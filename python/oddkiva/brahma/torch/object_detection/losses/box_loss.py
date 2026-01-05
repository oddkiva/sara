# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torch
import torch.nn.functional as F

from oddkiva.brahma.torch.object_detection.losses.box_losses import loss_giou


class BoxLoss(torch.nn.Module):

    def __init__(self,
                 eps: float = 1e-8,
                 w_l1: float = 1.0,
                 w_giou: float = 1.0):
        super(BoxLoss, self).__init__()
        self.eps = eps
        self.w_l1 = w_l1
        self.w_giou = w_giou

    def forward(
        self,
        query_boxes: torch.Tensor,
        target_boxes: list[torch.Tensor], 
        matching: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        qboxes = torch.cat([
            query_boxes[n, qixs_n]
            for n, (qixs_n, _) in enumerate(matching)
        ])

        tboxes = torch.cat([
            tboxes_n[tixs_n]
            for tboxes_n, (_, tixs_n) in zip(target_boxes, matching)
        ])

        return \
            self.w_l1 * F.l1_loss(qboxes, tboxes) + \
            self.w_giou * loss_giou(qboxes, tboxes, self.eps)
