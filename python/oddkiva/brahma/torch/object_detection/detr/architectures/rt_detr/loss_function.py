# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F

from oddkiva.brahma.torch.object_detection.losses.box_matcher import BoxMatcher
from oddkiva.brahma.torch.object_detection.losses.box_loss import BoxLoss
from oddkiva.brahma.torch.object_detection.losses.focal_loss import FocalLoss
from oddkiva.brahma.torch.object_detection.losses.varifocal_loss import VarifocalLoss
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    dn_detr.contrastive_denoising_group_generator import (
        ContrastiveDenoisingGroupGenerator
    )


class RTDETRLossFunction(nn.Module):
    """
    The composite Hungarian loss function used in RT-DETR v2.
    """

    def __init__(self,
                 weight_dict: dict[str, float],
                 alpha: float = 0.2,
                 gamma: float = 2.0,
                 num_classes: int = 80):
        """Create the criterion.

        Parameters:
            weight_dict:
                dict of weights for each loss.

            losses:
                list of all the losses to be applied. See get_loss for list of available losses.

            num_classes:
                The number of object categories, omitting the special non-object
                category
        """
        super().__init__()
        self.matcher = BoxMatcher(alpha=alpha, gamma=gamma)

        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.varifocal_loss = VarifocalLoss(alpha=alpha, gamma=gamma)
        self.box_loss = BoxLoss()

        self.losses: dict[str, nn.Module] = {
            'vf': self.varifocal_loss,
            'box': self.box_loss
        }

        self.num_classes = num_classes
        self.weight_dict = weight_dict

    def labeling_focal_loss(
        self,
        query_class_logits: torch.Tensor,
        target_labels: list[torch.Tensor],
        matching: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int
    ):
        loss = self.focal_loss.forward(query_class_logits, target_labels, matching)
        # loss = loss.mean(1).sum() * query_class_logits.shape[1] / num_boxes
        return loss

    def labeling_varifocal_loss(self,
                                query_boxes: torch.Tensor,
                                query_class_logits: torch.Tensor,
                                target_boxes: list[torch.Tensor],
                                target_labels: list[torch.Tensor],
                                matching: list[tuple[torch.Tensor, torch.Tensor]],
                                num_boxes: int):
        return self.vf_loss.forward(query_boxes, query_class_logits,
                                    target_boxes, target_labels,
                                    matching,
                                    num_boxes)

    def loss_boxes(self,
                   query_boxes: torch.Tensor,
                   target_boxes: torch.Tensor,
                   matching: list[tuple[torch.Tensor, torch.Tensor]],
                   num_boxes: int | None = None):
        self.box_loss.forward(query_boxes, target_boxes, matching)

    def count_targets(self, targets: list[torch.Tensor]) -> torch.Tensor:
        # Compute the average number of target boxes across all nodes, for
        # normalization purposes
        tcount = sum(len(t) for t in targets)
        tcount = torch.as_tensor([tcount], dtype=torch.float,
                                    device=targets[0].device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(tcount)
        tcount = torch.clamp(tcount / get_world_size(), min=1).item()

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
            'box': self.box_loss.forward(qboxes, tboxes, matching)
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
                dn_classs_logits: torch.Tensor,
                dn_groups: ContrastiveDenoisingGroupGenerator.Output,
                # The ground-truth data.
                target_boxes: list[torch.Tensor],
                target_labels: list[torch.Tensor]):
        """ This performs the loss computation.
        """
        assert len(query_boxes.shape) == 4  # (iterations, batch_size, top_K, 4)
        assert query_boxes.shape[1:] == anchor_boxes.shape
        assert query_class_logits.shape[1:] == anchor_class_logits.shape

        # Compute the average number of target boxes across all nodes, for
        # normalization purposes
        target_count = self.count_targets(target_labels)

        matching = self.matcher(outputs_without_aux, targets)

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
        losses_final = self.compute_loss_dict(qboxes_final, qlogits_final,
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
        losses_iterations = []
        for qboxes_i, qlogits_i in zip(query_boxes[:-1],
                                       query_class_logits[:-1]):
            matching_i = self.matcher.forward(qclass_logits_i, qboxes_i,
                                              target_labels, target_boxes)
            losses = self.compute_loss_dict(qboxes_i, qlogits_i,
                                            target_boxes, target_labels,
                                            matching_i, target_count)
            losses_iterations.append(losses)

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
            for (tgt_boxes_n, (_, tixs_n)) in zip(tgt_boxes, matching_dn)
        ]
        tgt_labels_dn = [
            tgt_labels_n[tixs_n]
            for (tgt_labels_n, (_, tixs_n)) in zip(tgt_labels, matching_dn)
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
        losses_anchors = self.compute_loss_dict(
            anchor_boxes, anchor_class_logits,
            target_boxes, target_labels,
            matching_anchors, target_count
        )

        return {
            'final': losses_final,
            'iters': losses_iterations,
            'init': losses_anchors,
            'dn': losses_dn
        }
