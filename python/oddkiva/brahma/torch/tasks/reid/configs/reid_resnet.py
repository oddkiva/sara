# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from collections import OrderedDict

import torch
import torchvision

from oddkiva.brahma.torch.backbone.resnet50 import ResNet50Variant


class ReidDescriptor50(torch.nn.Module):

    def __init__(self, dim: int = 256):
        super(ReidDescriptor50, self).__init__()
        self.resnet50_backbone = torch.nn.Sequential(
            *list(torchvision.models.resnet50().children())[:-1],
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(2048, dim),
            torch.nn.Softmax()
        )
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet50_backbone(x)



class ResNet50VariantReid(torch.nn.Module):
    def __init__(self, dim: int = 256):
        super(ResNet50VariantReid, self).__init__()
        self.dim = dim
        self.architecture = torch.nn.Sequential(
            OrderedDict([
                ('backbone', ResNet50Variant()),
                ('pool2d', torch.nn.AdaptiveAvgPool2d((1, 1))),
                ('flatten', torch.nn.Flatten(start_dim=1)),
                ('linear', torch.nn.Linear(2048, dim)),
                ('softmax', torch.nn.Softmax()),
            ])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.architecture(x)
        return y
