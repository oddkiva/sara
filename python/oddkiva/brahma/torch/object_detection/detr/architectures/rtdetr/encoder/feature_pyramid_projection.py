import torch
import torch.nn as nn

from oddkiva.brahma.torch.backbone.resnet.rtdetrv2_variant import (
    UnbiasedConvBNA
)


class FeaturePyramidProjection(nn.Module):

    def __init__(self, in_channels_list: list[int], out_channels: int):
        super().__init__()
        self.projections = nn.ModuleList([
            UnbiasedConvBNA(in_channels, out_channels, 1, 1,
                            activation=None)
            for in_channels in in_channels_list
        ])
        self._reinitialize_learning_parameters()

    def _reinitialize_learning_parameters(self):
        # if self.learn_query_content:
        #     nn.init.xavier_uniform_(self.tgt_embed.weight)

        # Reset the parameters
        for convbna in self.projections:
            assert type(convbna) is UnbiasedConvBNA
            conv = convbna.layers[0]
            assert type(conv) is nn.Conv2d
            nn.init.xavier_uniform_(conv.weight)


    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(xs) == len(self.projections)
        ys = [proj(x) for x, proj in zip(xs, self.projections)]
        return ys
