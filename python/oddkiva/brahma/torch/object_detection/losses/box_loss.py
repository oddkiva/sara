# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torch
import torch.nn.functional as F

from oddkiva.brahma.torch.object_detection.common.box_ops import (
    from_cxcywh_to_ltrb_format
)
from oddkiva.brahma.torch.object_detection.losses.box_losses import loss_giou


class BoxLoss(torch.nn.Module):

    def __init__(self, eps: float = 1e-8):
        super(BoxLoss, self).__init__()
        self.eps = eps

    def forward(
        self,
        query_boxes: torch.Tensor,
        target_boxes: list[torch.Tensor],
        matching: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int | None = None
    ) -> dict[str, torch.Tensor]:
        qboxes = torch.cat([
            query_boxes[n, qixs_n]
            for n, (qixs_n, _) in enumerate(matching)
        ])

        tboxes = torch.cat([
            tboxes_n[tixs_n]
            for tboxes_n, (_, tixs_n) in zip(target_boxes, matching)
        ])

        if num_boxes is None:
            num_boxes = sum([len(tgt_boxes_n) for tgt_boxes_n in target_boxes])

        l1_loss = F.l1_loss(qboxes, tboxes, reduction='sum') / num_boxes

        giou_loss = loss_giou(
            from_cxcywh_to_ltrb_format(qboxes),
            from_cxcywh_to_ltrb_format(tboxes),
            eps=self.eps,
            only_compute_diagonal=True
        ).sum() / num_boxes

        box_loss_dict = {
            'l1': l1_loss,
            'giou': giou_loss
        }

        return box_loss_dict
