from typing import Iterable

import torch

from oddkiva.brahma.torch.transformers.embedding.positional_sine_embedding \
    import (
        PositionalSineEmbedding2D
    )
from oddkiva.brahma.torch.backbone.repvgg import RepVggBaseLayer

class RepVggBlock(torch.nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int,
                 num_layers: int = 4, activation = 'silu'):
        self.layers = torch.nn.Sequential([
            RepVggBaseLayer(in_channels, hidden_channels, stride=1,
                            use_identity_connection=False,
                            activation=activation,
                            inplace_activation=False)
        ])
        for _ in range(num_layers - 1):
            self.layers.append(
                RepVggBaseLayer(hidden_channels, hidden_channels,
                                stride=1,
                                activation=activation,
                                inplace_activation=inplace_activation)
            )


class Fusion(torch.nn.Module):

    def __init__(self, rep_block_count: int):
        super().__init__()

    def forward(
        self,
        F_i: torch.Tensor, S_i_m_1: torch.Tensor
    ) -> torch.Tensor:
        pass


class CCFF(torch.nn.Module):
    """
    CCFF stands for (C)NN-based (C)ross-scale (F)eature (F)usion.

    The AIFI module improves the coarsest feature map (F5). The improved
    feature map is denoted as F5++. Then, CCFF will inject top-down the
    semantic object information contained in feature map F5++ to the feature map
    F4, and recursively to F3 and so on.
    """

    def __init__(self):
        super().__init__()

    def forward(self, S: Iterable[torch.Tensor]) -> torch.Tensor:
        pass
