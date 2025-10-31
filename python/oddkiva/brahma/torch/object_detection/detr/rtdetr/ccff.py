from typing import Iterable

import torch
import torch.nn.functional as F

from oddkiva.brahma.torch.backbone.resnet50 import ConvBNA
from oddkiva.brahma.torch.backbone.repvgg import RepVggBaseLayer

class RepVggBlock(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 layer_count: int = 4,
                 activation: str = 'silu'):
        super().__init__()
        self.layers = torch.nn.Sequential(*[
            RepVggBaseLayer(in_channels, out_channels, stride=1,
                            use_identity_connection=False,
                            activation=activation,
                            inplace_activation=False)
            for _ in range(layer_count)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Fusion(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_dim_expansion_factor: float = 1.0,
                 repvgg_layer_count: int = 3,
                 activation: str = 'silu'):
        super().__init__()
        hidden_dim = int(out_channels * hidden_dim_expansion_factor)
        self.conv1 = ConvBNA(in_channels, hidden_dim,
                             1, 1, True, activation, 1)
        self.conv2 = ConvBNA(in_channels, hidden_dim,
                             1, 1, True, activation, 1)
        self.repvgg_block = RepVggBlock(
            hidden_dim, hidden_dim,
            layer_count=repvgg_layer_count,
            activation=activation
        )
        if hidden_dim != out_channels:
            self.conv3 = ConvBNA(hidden_dim, out_channels,
                                 1, 1, True, activation, 3)
        else:
            self.conv3 = torch.nn.Identity()

    def forward(
        self,
        F: torch.Tensor,
        S: torch.Tensor
    ) -> torch.Tensor:
        FS = torch.cat((F, S), dim=1)
        FS1 = self.conv1(FS)
        FS2 = self.repvgg_block(self.conv2(FS))
        return FS1 + FS2


class YellowConvBlock(ConvBNA):
    """
    This class implements the yellow block in Figure 4:
    (Conv1x1 s1 -> BN -> SiLU).

    Basically, it reduces the dimension of the feature vectors.

    I don't know how to name them...
    """

    def __init__(self, in_channels: int, out_channels: int, id: int,
                 activation: str | None = 'silu'):
        super(YellowConvBlock, self).__init__(
            in_channels, out_channels,
            1, 1,        # Kernel size and stride
            True,        # Batch-normalization
            activation,  # Activation
            id           # ID
        )

class BlueConvBlock(ConvBNA):
    """
    This class implements the blue block in Figure 4:
    (Conv3x3 s2 -> BN -> SiLU).

    Basically, it downsamples (and also does more than that) the feature maps.

    I don't know how to name them...
    """

    def __init__(self, in_channels: int, out_channels: int, id: int,
                 activation: str | None = 'silu'):
        super(BlueConvBlock, self).__init__(
            in_channels, out_channels,
            3, 2,        # Kernel size and stride
            True,        # Batch-normalization
            activation,  # Activation
            id           # ID
        )


class CCFF(torch.nn.Module):
    """
    CCFF stands for (C)NN-based (C)ross-scale (F)eature (F)usion.

    The AIFI module improves the coarsest feature map (S5). The improved
    feature map is denoted as F5 and should be seen as a query map.

    Then, CCFF injects top-down the semantic object information contained
    in query map F5 to the feature map S4, and recursively to S3 and so on.
    Therefore, we produce query maps F4 and F3.
    - F5
    - F4 <-- enrich(F5, S4)
    - F3 <-- enrich(F4, S3)

    Finally, CCFF refines in a bottom-up the query maps:
    - F3++ <-- F3++
    - F4++ <-- refine(F3++, F4)
    - F5++ <-- refine(F4++, F5)

    We follow the implementation as detailed in Figure 4 of the paper.
    """

    def __init__(
        self,
        feature_dims: list[int],
        hidden_dim: int = 256,
    ):
        super().__init__()
        # Top-down semantic enrichment.
        self.yellow_blocks = torch.nn.ModuleList([
            YellowConvBlock(feature_dims[i], hidden_dim, i)
            for i in range(len(feature_dims) - 1)
        ])
        # Bottom-up refinement.
        self.blue_blocks = torch.nn.ModuleList([
            BlueConvBlock(feature_dims[i], hidden_dim, i)
            for i in range(len(feature_dims) - 1)
        ])
        self.fusions = torch.nn.ModuleList([
            Fusion(feature_dims[i], hidden_dim)
            for i in range(len(feature_dims))
        ])

    def enrich_topdown(self, F5: torch.Tensor, S: list[torch.Tensor]):
        """
        Enrich the finer-scale feature maps with semantic information, in a
        recursive manner.

        Parameters
        ----------
        F5:
            query as a feature map $(N, d_k, H, W)$

        S:
            the feature maps of the feature pyramid produced from the CNN
            backbone
        """

        query_maps_enriched_topdown = [F5]
        query_maps_yellowed = []

        for i in range(len(S) - 1, 0, -1):
            # Project the query maps into a lower dimensional space.
            query_maps_yellowed.append(
                self.yellow_blocks[i - 1](query_maps_enriched_topdown[-1])
            )
            # Upscale the coarse query map.
            query_map_yellowed_upscaled = F.interpolate(
                query_maps_yellowed[-1],
                scale_factor=2,
                mode='nearest'
            )
            # Imbue the semantic information to the finer feature map S[i - 1]
            # with a fusion operation.
            query_maps_enriched_topdown.append(
                self.fusions[i - 1](
                    query_map_yellowed_upscaled,
                    S[i - 1])
            )

        query_maps_yellowed.reverse()
        query_maps_enriched_topdown.reverse()

        return (query_maps_enriched_topdown, query_maps_yellowed)

    def refine_bottomup(
        self,
        query_maps_enriched_topdown: list[torch.Tensor],
        query_maps_yellowed: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        n = len(query_maps_enriched_topdown)

        # Bottom-up semantic refinement in a recursive fashion.
        query_maps_refined_bottomup = [query_maps_enriched_topdown[0]]
        query_maps_blued = []

        for i in range(1, n):
            # Downsample the finer-scale query maps.
            query_maps_blued.append(
                self.blue_blocks[i](query_maps_refined_bottomup[i])
            )
            # Fuse the finer-scale and next coarser-scale query maps.
            # This should refine the coarser query maps.
            query_maps_refined_bottomup.append(
                self.fusions[i](
                    query_maps_blued[i],
                    query_maps_yellowed[i])
            )

        return query_maps_refined_bottomup

    def forward(
        self,
        F5: torch.Tensor,
        S: list[torch.Tensor]
    ) -> torch.Tensor:
        (query_maps_enriched_topdown,
         query_maps_yellowed) = self.enrich_topdown(F5, S)

        query_maps_refined_bottomup = self.refine_bottomup(
            query_maps_enriched_topdown,
            query_maps_yellowed
        )

        return torch.cat(query_maps_refined_bottomup, dim=1)\
            .flatten(2)\
            .permute(0, 2, 1)
