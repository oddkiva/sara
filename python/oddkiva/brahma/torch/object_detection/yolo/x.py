# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from collections import OrderedDict

import torch
import torch.nn as nn


class YOLOX(nn.Module):

    def __init__(self, embed_dim: int, num_classes: int):
        super(YOLOX, self).__init__()

        # Object probability prediction head
        self.object_head = nn.Sequential(
            OrderedDict([
                ("obj-linear", nn.Linear(embed_dim, 1)),
                ("obj-prob", nn.Sigmoid()),
            ])
        )

        # Object class probability prediction head
        self.object_classes_head = nn.Sequential(
            OrderedDict([
                ("obj-class-linear", nn.Linear(embed_dim, num_classes)),
                ("obj-class-softmax", nn.Softmax(dim=-1)),
            ])
        )

        # Box geometry neck
        self.box_geometry_neck = nn.Sequential(
            OrderedDict([
                ("obj-class-linear", nn.Linear(embed_dim, 4)),
                ("obj-class-softmax", nn.Sigmoid()),
            ])
        )

        # Box center head.
        self.box_center_head = nn.Sequential(
            OrderedDict([
                ("box-center-linear", nn.Linear(embed_dim, 2)),
                ("box-center-activation", nn.Sigmoid()),
            ])
        )

        # Box size head.
        self.box_sizes_head = nn.Sequential(
            OrderedDict([
                ("box-sizes-linear", nn.Linear(embed_dim, 2)),
                ("box-sizes-activation", nn.Sigmoid()),
            ])
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        object_probs = self.object_head(x)
        object_class_probs = self.object_classes_head(x)

        box_embeds = self.box_geometry_neck(x)
        box_centers = self.box_center_head(box_embeds[:, :, :2])
        box_sizes = self.box_sizes_head(box_embeds[:, :, 2:])
        box_predictions = torch.cat((box_centers, box_sizes), dim=-1)

        return {
            'object_probs': object_probs,
            'object_class_probs': object_class_probs,
            'boxes_predictions': box_predictions
        }
