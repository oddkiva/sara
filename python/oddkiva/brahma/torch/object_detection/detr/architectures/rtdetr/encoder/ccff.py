from collections import OrderedDict

import torch
import torch.nn.functional as F

from oddkiva.brahma.torch.backbone.resnet.rtdetrv2_variant import UnbiasedConvBNA
from oddkiva.brahma.torch.backbone.repvgg import RepVggBlock


class Fusion(torch.nn.Module):
    """
    The `Fusion` class implements the building block of the Path-Aggregation
    architectural scheme described in the PANet paper.

    It is used to recursively flow the semantic information. The fusion block
    is used in two passes.

    1. The first pass is top-down, where the semantic information flows from
        coarse feature maps to finer feature maps.
    2. The second pass is bottom-up manner, where fine feature maps, which are
       now enriched with semantic information flows back the improved
       information to coarse feature maps.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_dim_expansion_factor: float = 1.0,
                 repvgg_layer_count: int = 3,
                 activation: str = 'silu'):
        super().__init__()
        hidden_dim = int(out_channels * hidden_dim_expansion_factor)
        self.conv1 = UnbiasedConvBNA(in_channels, hidden_dim, 1, 1,
                                     1, activation=activation)
        self.conv2 = UnbiasedConvBNA(in_channels, hidden_dim, 1, 1,
                                     2, activation=activation)
        self.repvgg_block = RepVggBlock(
            hidden_dim, hidden_dim,
            layer_count=repvgg_layer_count,
            activation=activation
        )
        if hidden_dim != out_channels:
            self.conv3 = UnbiasedConvBNA(hidden_dim, out_channels, 1, 1,
                                         3, activation=activation)
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


class LateralConvolution(UnbiasedConvBNA):
    """
    This class implements the yellow convolutional block in Figure 4 of the
    paper: (Conv1x1 s1 -> BN -> SiLU).
    The authors call it *lateral convolution* in the original code.

    Basically, it reduces the dimension of the feature vectors via a linear
    projection ("channel projection" in the original code of the
    `HybridEfficientEncoder` class).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels, out_channels, 1, 1, 0, activation='silu')


class DownsampleConvolution(UnbiasedConvBNA):
    """
    This class implements the blue block in Figure 4 of the paper:
    (Conv3x3 s2 -> BN -> SiLU).

    The authors did not make a class for that but I do it for the sake of code
    clarity. Please refer to the original `HybridEfficientEncoder` class, where
    they add the class member `self.downsample_convs`.
    """

    def __init__(self, in_channels: int, out_channels: int, id: int,
                 activation: str | None = 'silu'):
        super().__init__(
            in_channels, out_channels, 3, 2,  # Kernel size and stride
            id,                     # ID
            activation=activation,  # Activation
        )


class CCFF(torch.nn.Module):
    r"""
    CCFF stands for (C)NN-based (C)ross-scale (F)eature (F)usion.

    The AIFI module improves the coarsest feature map $\mathbf{S}_5$ of the CNN
    backbone. The improved feature map is denoted as $\mathbf{F}_5$ and should
    be seen as a query map.

    Then, CCFF injects top-down the semantic object information contained in
    query map $\mathbf{F}_5$ to the feature map $\mathbf{S}_4$, and recursively
    to $\mathbf{S}_3$ and so on. Therefore, we produce query maps
    $\mathbf{F}_4$ and $\mathbf{F}_3$.

    - $\mathbf{F}_5$
    - $\mathbf{F}_4 \leftarrow \mathrm{enrich}(\mathbf{F}_5, \mathbf{S}_4)$
    - $\mathbf{F}_3 \leftarrow \mathrm{enrich}(\mathbf{F}_4, \mathbf{S}_3)$

    Finally, CCFF refines in a bottom-up the query maps:

    - $\mathbf{F}_3^{++} \leftarrow \mathbf{F}_3^{++}$
    - $\mathbf{F}_4^{++} \leftarrow \mathrm{refine}(\mathbf{F}_3^{++}, \mathbf{F}_4)$
    - $\mathbf{F}_5^{++} \leftarrow \mathrm{refine}(\mathbf{F}_4^{++}, \mathbf{F}_5)$

    We follow the implementation as detailed in Figure 4 of the paper.
    """

    def __init__(
        self,
        feature_dims: list[int],
        hidden_dim: int = 256,
    ):
        super().__init__()
        # Top-down semantic enrichment.
        self.lateral_convs = torch.nn.ModuleList([
            LateralConvolution(feature_dims[i], hidden_dim)
            for i in range(len(feature_dims) - 1)
        ])
        self.top_down_fusion_blocks = torch.nn.ModuleList([
            Fusion(feature_dims[i], hidden_dim)
            for i in range(len(feature_dims) - 1)
        ])

        # Bottom-up refinement.
        self.downscale_convs = torch.nn.ModuleList([
            DownsampleConvolution(feature_dims[i], hidden_dim, i)
            for i in range(len(feature_dims) - 1)
        ])
        self.bottom_up_fusion_blocks = torch.nn.ModuleList([
            Fusion(feature_dims[i], hidden_dim)
            for i in range(len(feature_dims) - 1)
        ])

    def upscale(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode='nearest')

    def enrich_topdown(
        self,
        F5: torch.Tensor,
        S: list[torch.Tensor]
    ): #-> tuple(list[torch.Tensor], list[torch.Tensor]):
        """
        Enrich the finer-scale feature maps with semantic information, in a
        recursive manner.

        Notice that I use peculiar colored references, to help me find out what refers to
        what in the code and in the paper.

        Parameters:
            F5:
                the query matrix as a feature map $(N, d_k, H, W)$
            S:
                the feature maps of the feature pyramid produced from the CNN
                backbone
        """

        query_maps_enriched_topdown = [F5]
        query_maps_yellowed = []

        for i in range(len(S) - 1, 0, -1):
            # Project the query maps into a lower dimensional space.
            query_maps_yellowed.append(
                self.lateral_convs[i - 1](query_maps_enriched_topdown[-1])
            )

            # Upscale the coarse query map.
            query_map_yellowed_upscaled = self.upscale(query_maps_yellowed[-1])

            # Imbue the semantic information to the finer feature map S[i - 1]
            # with a fusion operation.
            query_maps_enriched_topdown.append(
                self.top_down_fusion_blocks[i](
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
                self.downscale_convs[i](query_maps_refined_bottomup[i])
            )
            # Fuse the finer-scale and next coarser-scale query maps.
            # This should refine the coarser query maps.
            query_maps_refined_bottomup.append(
                self.bottom_up_fusion_blocks[i - 1](
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
