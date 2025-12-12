from typing import Any
from dataclasses import astuple, dataclass

import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Box operations
# ----------------------------------------------------------------------
def from_cxcywh_to_ltrb_box_format(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    l = cx - 0.5 * w
    r = cx + 0.5 * w
    t = cy - 0.5 * h
    b = cy + 0.5 * h
    return torch.stack((l, t, r, b), dim=-1)


def fix_ltrb_boxes(boxes: torch.Tensor) -> torch.Tensor:
    l = boxes[..., 0]
    t = boxes[..., 1]
    r = boxes[..., 2]
    b = boxes[..., 3]
    boxes_fixed = torch.zeros_like(boxes)
    boxes_fixed[..., 0] = torch.where(l <= r, l, r)
    boxes_fixed[..., 1] = torch.where(l > r, r, l)
    boxes_fixed[..., 2] = torch.where(t <= b, t, b)
    boxes_fixed[..., 3] = torch.where(t > b, b, t)
    return boxes_fixed


def from_ltrb_to_cxcywh_box_format(boxes: torch.Tensor) -> torch.Tensor:
    l, t, r, b = boxes.unbind(-1)
    cx = 0.5 * (l + r)
    cy = 0.5 * (t + b)
    w = r - l
    h = t + b
    return torch.stack((cx, cy, w, h), dim=-1)


# ----------------------------------------------------------------------
# Math operations.
# ----------------------------------------------------------------------
def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5):
    x = x.clip(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


# ----------------------------------------------------------------------
# ContrastiveDenoisingGroupGenerator
# ----------------------------------------------------------------------
class ContrastiveDenoisingGroupGenerator(nn.Module):
    """The `ContrastiveDenoisingGroupGenerator` constructs groups of perturbed
    ground-truth object boxes as described in the paper

    It is called:
    - *contrastive* because it generates positive samples and negative samples.

      For each ground-truth labeled box, we generate two sets of samples.
      - Positive samples are those which are very close to this ground-truth
        box and which overlap strongly with it.
      - Negative samples are those which overlap very little with it. Now
        specifically, they are generated in such a way that they are actually
        located at the "periphery" of the ground-truth samples, without

    - *denoising* because they are noised ground-truth samples meant to be
      denoised.

    [DN-DETR: Accelerate DETR Training by Introducing Query
    DeNoising](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_DN-DETR_Accelerate_DETR_Training_by_Introducing_Query_DeNoising_CVPR_2022_paper.pdf)

    This is an important module that makes the training of DETR-based
    architectures a lot faster as demonstrated in the paper.
    """

    @dataclass
    class StackedBoxAnnotations:
        labels: torch.Tensor
        geometries: torch.Tensor
        padding_mask: torch.Tensor

    @dataclass
    class Output:
        box_labels: torch.Tensor
        box_geometries: torch.Tensor
        attention_mask: torch.Tensor
        dn_meta: dict[str, Any]

    def __init__(self,
                 class_count: int,
                 shared_box_class_embedding: nn.Embedding,
                 box_count: int = 100,
                 box_label_flip_probability: float = 0.5,
                 box_noise_relative_scale: float = 1.0):
        super().__init__()

        self.class_count = class_count
        self.shared_box_class_embedding = shared_box_class_embedding
        self.box_label_flip_probability = box_label_flip_probability
        self.box_count = box_count
        self.ltrb_noise_rel_magnitude = box_noise_relative_scale

    def stack_labeled_boxes(
        self,
        box_annotations: dict[str, list[torch.Tensor]],
        num_classes: int,
    ) -> StackedBoxAnnotations:
        boxes = box_annotations['boxes']
        labels = box_annotations['labels']
        # Let us impose the following hard constraint:
        # -> no empty image annotations, please!
        assert len(boxes) == len(labels)
        assert all(len(b) > 0 for b in boxes) is True
        assert all(len(b) == len(l) for (b, l) in zip(boxes, labels)) is True

        # What we do is simply tensorize the ground truth data,
        # with appropriate padding.
        box_count_max = max([len(b) for b in boxes])
        batch_size = len(boxes)

        # Stack the list of tensors and appropriately pad each tensors
        stacked_labels = torch.full(
            [batch_size, box_count_max],
            num_classes,
            dtype=torch.int32,
            device=boxes[0].device
        )
        stacked_geometries = torch.zeros(
            [batch_size, box_count_max, 4],
            device=boxes[0].device
        )
        padding_mask = torch.zeros(
            [batch_size, box_count_max],
            dtype=torch.bool,
            device=boxes[0].device
        )

        for n in range(batch_size):
            box_count = len(boxes[n])
            stacked_labels[n, :box_count] = labels[n]
            stacked_geometries[n, :box_count] = boxes[n]
            padding_mask[n, :box_count] = True

        return self.StackedBoxAnnotations(
            stacked_labels,
            stacked_geometries,
            padding_mask
        )

    def perturb_box_labels(
        self,
        rep_box_labels: torch.Tensor,
        rep_box_mask: torch.Tensor
    ) -> torch.Tensor:
        assert \
            0.0 < self.box_label_flip_probability and\
            self.box_label_flip_probability < 1.0

        p_perturb = self.box_label_flip_probability * 0.5
        uni01_samples = torch.rand_like(rep_box_labels)
        mask_perturb = uni01_samples < p_perturb

        wrong_labels = torch.randint_like(rep_box_labels,
                                          low=0,
                                          high=self.class_count,
                                          dtype=rep_box_labels.dtype)
        rep_labels_perturbed = torch.where(
            mask_perturb & rep_box_mask, wrong_labels, rep_box_labels
        )

        return rep_labels_perturbed

    def perturb_box_geometries(self, rep_box_geometries: torch.Tensor,
                               rep_box_mask: torch.Tensor,
                               N: int, B: int, G: int):
        device = rep_box_geometries.device
        # Generate positive boxes and negative boxes for each ground-truth
        # box.
        # - A positive box is a perturbed ground-truth box whose geometry
        #   should be similar to the original ground-truth box geometry.
        # - A negative box is a perturbed ground-truth box whose geometry
        #   should have little similarity with the original ground-truth box.
        #   A negative box can still overlap with the original ground-truth
        #   box, but it should not be too far from the original one, so that we
        #   learn to effectively denoise perturbed positive ground-truth boxes.

        negative = torch.zeros((N, 2 * B, 1), device=device)
        negative[:, B:] = 1
        negative = negative.tile((1, G, 1))
        positive = 1 - negative

        # shape is (N, 2 * G, 1)
        positive = positive.squeeze(-1) * rep_box_mask
        positive_ixs = torch.nonzero(positive)
        ltrb_noise_rel_mag = 0.5 * self.ltrb_noise_rel_magnitude
        rep_box_whs = rep_box_geometries[:, :, 2:]
        ltrb_noise_mag_max = rep_box_whs.tile((1, 1, 2))
        noise_magnitudes = ltrb_noise_rel_mag * ltrb_noise_mag_max

        zero_or_one_samples = torch.randint_like(rep_box_geometries, 0, 2)
        noise_signs = zero_or_one_samples * 2 - 1  # -1 or +1

        # ----------------------------------------------------------------------
        # Randomly perturb the box geometries.
        #
        # A POSITIVE perturbed ground-truth box (cx, cy, w, h):
        # l <-- l + ɛ * 0.5 * x_l * w
        # r <-- r + ɛ * 0.5 * x_r * w
        # t <-- t + ɛ * 0.5 * x_t * h
        # b <-- b + ɛ * 0.5 * x_b * h
        #
        # A NEGATIVE perturbed ground-truth box:
        # l <-- l + ɛ * (1 + 0.5 * x_l) * w
        # r <-- r + ɛ * (1 + 0.5 * x_r) * w
        # t <-- t + ɛ * (1 + 0.5 * x_t) * w
        # b <-- b + ɛ * (1 + 0.5 * x_b) * h
        #
        # where:          ɛ ~ U({-1, +1})
        #       x_{l,r,t,b} ~ U([0, 1])
        #
        # This problem however is that the original implementation, as I
        # understand it, does not guarantee that the perturbed form is
        # well-formed. We don't have the guarantee that:
        # - l < r
        # - t < b
        # Which can lead to negative sizes, and that would be a shame.
        # ----------------------------------------------------------------------
        uni01_samples = torch.rand_like(rep_box_geometries)
        neg_additive_noise = (uni01_samples + 1.0) * negative
        pos_additive_noise = uni01_samples * (1 - negative)
        additive_noise_normalized = neg_additive_noise + pos_additive_noise

        additive_noise = noise_magnitudes * noise_signs * additive_noise_normalized

        boxes_noised = from_cxcywh_to_ltrb_box_format(rep_box_geometries)
        boxes_noised = fix_ltrb_boxes(boxes_noised + additive_noise)
        boxes_noised = boxes_noised.clip(min=0.0, max=1.0)
        boxes_noised = from_ltrb_to_cxcywh_box_format(boxes_noised)

        return boxes_noised, positive_ixs

    def forward(
        self,
        query_count: int,
        ground_truth: dict[str, list[torch.Tensor]]
    ) -> Output:
        boxes = ground_truth['boxes']
        labels = ground_truth['labels']
        assert len(boxes) == len(labels)

        box_count_max = max([len(b) for b in boxes])

        # The number of so-called `denoising groups` in DN-DETR.
        dn_group_count = self.box_count // box_count_max
        if dn_group_count == 0:
            dn_group_count = 1

        box_labels, box_geometries, box_pad_mask = astuple(self.stack_labeled_boxes(
            ground_truth,
            self.class_count
        ))

        N = len(boxes)
        G = dn_group_count
        B = box_count_max
        device = boxes[0].device

        rep_box_labels = box_labels.tile((1, 2 * G))
        # Shape is (N, B * 2 * G)

        rep_box_geometries = box_geometries.tile((1, 2 * G, 1))
        # Shape is (N, B * 2 * G, 4)

        rep_box_pad_mask = box_pad_mask.tile((1, 2 * G))
        # Shape is (N, B * 2 * G, 1)


        # ----------------------------------------------------------------------
        # Generate the so-called `denoising` groups of ground-truth boxes as
        # explained in DN-DETR.
        # ----------------------------------------------------------------------
        dn_labels = self.perturb_box_labels(rep_box_labels, rep_box_pad_mask)
        dn_geometries, positive_ixs = self.perturb_box_geometries(
            rep_box_geometries, rep_box_pad_mask, N, B, G
        )
        dn_geometry_logits = inverse_sigmoid(dn_geometries)

        # ----------------------------------------------------------------------
        # Build the attention mask as explained in DN-DETR.
        # ----------------------------------------------------------------------
        dn_box_count = B * 2 * G
        all_query_count = query_count + dn_box_count
        attn_mask = torch.full((all_query_count, all_query_count), False,
                               dtype=torch.bool,
                               device=device)

        # Any box from group `i` cannot interact with any box from group `j`,
        # for any group pairs `(i, j)`, with `i != j`.
        B2 = B * 2
        BG2 = dn_box_count
        for g in range(dn_group_count):
            if g == 0:
                attn_mask[0:B2, B2:BG2] = True
            if g == dn_group_count - 1:
                attn_mask[B2*g:B2*(g + 1), :B2*g] = True
            else:
                attn_mask[B2*g:B2*(g + 1), B2*(g + 1):BG2] = True
                attn_mask[B2*g:B2*(g + 1), :B2*g] = True

        return self.Output(
            dn_labels,
            dn_geometry_logits,
            attn_mask,
            {
                'dn_positive_ixs': positive_ixs,
                'dn_group_count': dn_group_count,
                'dn_num_splits': (dn_box_count, self.query_count)
            }
        )
