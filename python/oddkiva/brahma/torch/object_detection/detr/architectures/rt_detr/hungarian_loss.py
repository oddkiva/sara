# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from typing import Any

import torch
import torch.nn as nn
from torch.distributed import ReduceOp
from torch.utils.tensorboard.writer import SummaryWriter

from oddkiva.brahma.torch.object_detection.losses.box_matcher import BoxMatcher
from oddkiva.brahma.torch.object_detection.losses.box_loss import BoxLoss
from oddkiva.brahma.torch.object_detection.losses.focal_loss import FocalLoss
from oddkiva.brahma.torch.object_detection.losses.varifocal_loss import VarifocalLoss
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    dn_detr.contrastive_denoising_group_generator import (
        ContrastiveDenoisingGroupGenerator
    )
from oddkiva.brahma.torch.parallel.ddp import (
    get_world_size,
    is_ddp_available_and_initialized
)


class RTDETRHungarianLoss(nn.Module):
    """
    The composite Hungarian loss function used in RT-DETR v2.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 2.0,
        weights: dict[str, float] = {
            'class': 2.0,
            'l1': 5.0,
            'giou': 2.0
        }
    ):
        """Initializes the Hungarian loss function.
        """
        super().__init__()
        self.matcher = BoxMatcher(alpha=alpha, gamma=gamma, weights=weights)

        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.varifocal_loss = VarifocalLoss(alpha=alpha, gamma=gamma)
        self.box_loss = BoxLoss()

    def labeling_focal_loss(
        self,
        query_class_logits: torch.Tensor,
        target_labels: list[torch.Tensor],
        matching: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int | None = None
    ):
        loss = self.focal_loss.forward(query_class_logits, target_labels, matching)

        if num_boxes is None:
            num_boxes = sum([len(l) for l in target_labels])

        loss = loss.mean(1).sum() * query_class_logits.shape[1] / num_boxes
        return loss

    def labeling_varifocal_loss(self,
                                query_boxes: torch.Tensor,
                                query_class_logits: torch.Tensor,
                                target_boxes: list[torch.Tensor],
                                target_labels: list[torch.Tensor],
                                matching: list[tuple[torch.Tensor, torch.Tensor]],
                                num_boxes: int | None = None):
        return self.varifocal_loss.forward(
            query_boxes, query_class_logits,
            target_boxes, target_labels,
            matching,
            num_boxes
        )

    def loss_boxes(self,
                   query_boxes: torch.Tensor,
                   target_boxes: list[torch.Tensor],
                   matching: list[tuple[torch.Tensor, torch.Tensor]],
                   num_boxes: int | None = None):
        self.box_loss.forward(query_boxes, target_boxes, matching, num_boxes)

    def count_targets(self, targets: list[torch.Tensor]) -> int | float:
        # Compute the average number of target boxes across all nodes, for
        # normalization purposes
        tcount = sum(len(t) for t in targets)
        tcount = torch.as_tensor([tcount],
                                 dtype=torch.float,
                                 device=targets[0].device)
        if is_ddp_available_and_initialized():
            torch.distributed.all_reduce(tcount)
        tcount = torch.clamp(tcount / get_world_size(), min=1).item()
        return tcount

    def compute_loss_dict(self,
                          qboxes, qlogits,
                          tboxes, tlabels,
                          matching,
                          num_boxes) -> dict[str, torch.Tensor]:
        return {
            'vf': self.varifocal_loss.forward(qboxes, qlogits,
                                              tboxes, tlabels,
                                              matching,
                                              num_boxes),
            **self.box_loss.forward(qboxes, tboxes, matching)
        }

    def forward(self,
                # The final box predictions.
                query_boxes: torch.Tensor,
                query_class_logits: torch.Tensor,
                # Intermediate training outputs.
                anchor_boxes: torch.Tensor,
                anchor_class_logits: torch.Tensor,
                # Auxiliary denoising outputs.
                dn_boxes: torch.Tensor,
                dn_class_logits: torch.Tensor,
                dn_groups: ContrastiveDenoisingGroupGenerator.Output,
                # The ground-truth data.
                target_boxes: list[torch.Tensor],
                target_labels: list[torch.Tensor]):
        """ Calculates the linear combination of several Hungarian losses used
        in RT-DETR.
        """
        assert len(query_boxes.shape) == 4  # (iterations, batch_size, top_K, 4)
        assert query_boxes.shape[1:] == anchor_boxes.shape
        assert query_class_logits.shape[1:] == anchor_class_logits.shape

        # Compute the average number of target boxes across all nodes, for
        # normalization purposes
        target_count = self.count_targets(target_labels)

        # Optimize:
        # - the transformer decoder using the FINAL predictions.
        # - the backbone+hybrid encoder is also optimized because we feed the
        #   memory tensor.
        #   The memory tensor is the flattened feature pyramid produced by the
        #   backbone and the hybrid encoder.
        #
        # We must minimize the errors in the box geometries and their object
        # class probability vectors *at each iteration* of the refinement
        qboxes_final = query_boxes[-1]
        qlogits_final = query_class_logits[-1]
        matching = self.matcher.forward(qlogits_final, qboxes_final,
                                        target_labels, target_boxes)
        loss_final = self.compute_loss_dict(qboxes_final, qlogits_final,
                                               target_boxes, target_labels,
                                               matching, target_count)

        # Specifically optimize *each layer* of the transformer decoder using
        # *EACH ITERATION* towards the final predictions. This is to accelerate
        # the training convergence.
        #
        # We must minimize the errors in the box geometries and their object
        # class probability vectors *at each iteration* of the refinement
        #
        # Again note that the backbone+hybrid encoder is also optimized because
        # we feed the memory tensor in the transformer decoder.
        loss_iterations = []
        for qboxes_i, qlogits_i in zip(query_boxes[:-1],
                                       query_class_logits[:-1]):
            matching_i = self.matcher.forward(qlogits_i, qboxes_i,
                                              target_labels, target_boxes)
            losses = self.compute_loss_dict(qboxes_i, qlogits_i,
                                            target_boxes, target_labels,
                                            matching_i, target_count)
            loss_iterations.append(losses)


        # Optimize:
        # 1. The transformer decoder network.
        # 2. the denoising embedding vector space using the denoising
        #    groups, which is only used at the training stage.
        #
        # The denoising process must minimize the errors in the box geometries
        # and the object class probability vectors.
        #
        # Likewise that the backbone+hybrid encoder is also optimized because
        # we feed the memory tensor in the transformer decoder.
        matching_dn = dn_groups.populate_matching(target_labels)
        tgt_boxes_dn = [
            tgt_boxes_n[tixs_n]
            for (tgt_boxes_n, (_, tixs_n)) in zip(target_boxes, matching_dn)
        ]
        tgt_labels_dn = [
            tgt_labels_n[tixs_n]
            for (tgt_labels_n, (_, tixs_n)) in zip(target_labels, matching_dn)
        ]
        losses_dn = [
            self.compute_loss_dict(dn_boxes_i, dn_class_logits_i,
                                   tgt_boxes_dn, tgt_labels_dn,
                                   matching_dn, target_count)
            for dn_boxes_i, dn_class_logits_i in zip(dn_boxes, dn_class_logits)
        ]

        # Optimize:
        # 1. the backbone, i.e., the PA-FPN -> HybridEncoder (AIFI+CCFF),
        #    which generates the anchor queries
        # 2. The class logit network which is used to rank and select the top-K
        #    anchor queries.
        #
        # The optimization consists in minimizing the errors in the box
        # geometries and the object class probability vectors.
        matching_anchors = self.matcher.forward(
            anchor_class_logits, anchor_boxes,
            target_labels, target_boxes
        )
        loss_anchors = self.compute_loss_dict(
            anchor_boxes, anchor_class_logits,
            target_boxes, target_labels,
            matching_anchors, target_count
        )

        return {
            'final': loss_final,
            'iters': loss_iterations,
            'init': loss_anchors,
            'dn': losses_dn
        }


class HungarianLossReducer(nn.Module):

    def __init__(self, weights: dict[str, float]):
        super().__init__()
        self.weights = weights

    def reduce_loss_dict(self, loss_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([
            self.weights[k] * loss_dict[k] for k in loss_dict
        ]).sum()


    def forward(self, loss_dict: dict[str, Any]) -> torch.Tensor:
        loss_final = loss_dict['final']
        loss_iterations = loss_dict['iters']
        loss_anchors = loss_dict['init']
        loss_dn = loss_dict['dn']

        loss_final = self.reduce_loss_dict(loss_final)

        loss_iterations = torch.stack([
            self.reduce_loss_dict(loss_i)
            for loss_i in loss_iterations
        ]).sum()

        loss_anchors = self.reduce_loss_dict(loss_anchors)

        loss_dn = torch.stack([
            self.reduce_loss_dict(losses_dn_i)
            for losses_dn_i in loss_dn
        ]).sum()

        return loss_final + loss_iterations + loss_dn + loss_anchors


def compute_ddp_average_loss_dict(loss_dict: dict[str, torch.Tensor]):
    avg_loss_values = torch.stack([loss_dict[k] for k in loss_dict]).sum()
    torch.distributed.all_reduce(avg_loss_values, op=ReduceOp.AVG)
    return avg_loss_values


def log_elementary_losses(loss_dict: dict[str, Any],
                          writer: SummaryWriter,
                          train_global_step: int) -> None:

    # Compute the average final loss value across all GPUs.
    loss_final = loss_dict['final']
    keys = [*loss_final.keys()]
    loss_values_f = compute_ddp_average_loss_dict(loss_final)
    # Log.
    for k, loss_value_f in zip(keys, loss_values_f):
        writer.add_scalar(f'train/loss/final/{k}', loss_value_f, train_global_step)

    # Compute the average iterated loss values across all GPUs.
    loss_iters = loss_dict['iters']
    for loss_iters_i in loss_iters:
        # Compute the average iterated loss value across all GPUs.
        keys = [*loss_iters_i.keys()]
        loss_values_i = compute_ddp_average_loss_dict(loss_iters_i)
        # Log.
        for k, loss_value_i in zip(keys, loss_values_i):
            writer.add_scalar(f'train/loss/iterated/{k}', loss_value_i, train_global_step)

    # Compute the average anchor loss value across all GPUs.
    loss_anchors = loss_dict['init']
    keys = [*loss_anchors.keys()]
    loss_values_a = compute_ddp_average_loss_dict(loss_anchors)
    # Log.
    for k, loss_value_a in zip(keys, loss_values_a):
        writer.add_scalar(f'train/loss/anchors/{k}', loss_value_a, train_global_step)

    # Compute the average denoised loss value across all GPUs.
    loss_dn = loss_dict['dn']
    for loss_dn_i in loss_dn:
        # Compute the average denoised loss value across all GPUs.
        keys = [*loss_dn_i.keys()]
        loss_values_dn_i = compute_ddp_average_loss_dict(loss_dn_i)
        # Log.
        for k, loss_value_i in zip(keys, loss_values_dn_i):
            writer.add_scalar(f'train/loss/dn/{k}', loss_value_i, train_global_step)
