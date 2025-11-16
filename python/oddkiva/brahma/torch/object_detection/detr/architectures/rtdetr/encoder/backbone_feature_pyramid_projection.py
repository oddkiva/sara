import torch
import torch.nn as nn

from oddkiva.brahma.torch.backbone.resnet.rtdetrv2_variant import (
    UnbiasedConvBNA
)


class BackboneFeaturePyramidProjection(nn.Module):

    def __init__(self, in_channels_list: list[int], out_channels: int):
        super().__init__()
        self.projections = nn.ModuleList([
            UnbiasedConvBNA(in_channels, out_channels, 1, 1,
                            activation=None)
            for in_channels in in_channels_list
        ])

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(xs) == len(self.projections)
        ys = [proj(x) for x, proj in zip(xs, self.projections)]
        return ys
