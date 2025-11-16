import torch
import torch.nn.functional as F

from oddkiva.brahma.torch.backbone.resnet.rtdetrv2_variant import UnbiasedConvBNA
from oddkiva.brahma.torch.backbone.repvgg import RepVggStack


class FusionBlock(torch.nn.Module):
    """
    The `FusionBlock` class implements the building block of the
    Path-Aggregation architectural scheme described in the PANet paper.

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
                                     id=1, activation=activation)
        self.conv2 = UnbiasedConvBNA(in_channels, hidden_dim, 1, 1,
                                     id=2, activation=activation)
        self.repvgg_stack = RepVggStack(
            hidden_dim, hidden_dim,
            layer_count=repvgg_layer_count,
            activation=activation
        )
        if hidden_dim != out_channels:
            self.conv3 = UnbiasedConvBNA(hidden_dim, out_channels, 1, 1,
                                         id=3, activation=activation)
        else:
            self.conv3 = torch.nn.Identity()

    def forward(
        self,
        F: torch.Tensor,
        S: torch.Tensor
    ) -> torch.Tensor:
        assert len(F.shape) == 4
        assert len(S.shape) == 4
        F_dim = F.shape[1]
        S_dim = F.shape[1]
        assert F_dim == S_dim

        FS = torch.cat((F, S), dim=1)
        FS1 = self.repvgg_stack(self.conv1(FS))
        FS2 = self.conv2(FS)
        return self.conv3(FS1 + FS2)


class LateralConvolution(UnbiasedConvBNA):
    r"""
    This class implements the yellow convolutional block in Figure 4 of the
    paper:
    ($\mathrm{conv}_{1\times1, s=1} \rightarrow \mathrm{BN} \rightarrow \mathrm{SiLU}$).
    The authors call it *lateral convolution* in the original code.

    Basically, it reduces the dimension of the feature vectors via a linear
    projection ("channel projection" in the original code of the
    `HybridEfficientEncoder` class).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels, out_channels, 1, 1, activation='silu')


class DownsampleConvolution(UnbiasedConvBNA):
    r"""
    This class implements the blue block in Figure 4 of the paper:
    ($\mathrm{conv}_{3\times3, s=2} \rightarrow \mathrm{BN} \rightarrow \mathrm{SiLU}$).

    The authors did not make a class for that but I do it for the sake of code
    clarity. Please refer to the original `HybridEncoder` class, where
    they add the class member `self.downsample_convs`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str | None = 'silu'
    ):
        super().__init__(
            in_channels, out_channels,
            3, 2,  # Kernel size and stride
            activation=activation,  # Activation
        )


class TopDownFusionNet(torch.nn.Module):
    """
    Top-down fusion network.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stack_count: int,
        hidden_dim_expansion_factor: float = 1.0,
        repvgg_stack_depth: int = 3,
        activation: str = 'silu'
    ):
        super().__init__()
        self.lateral_convs = torch.nn.ModuleList([
            LateralConvolution(out_channels, out_channels)
            for _ in range(stack_count)
        ])
        self.fusion_blocks = torch.nn.ModuleList([
            FusionBlock(in_channels, out_channels,
                   hidden_dim_expansion_factor=hidden_dim_expansion_factor,
                   repvgg_layer_count=repvgg_stack_depth,
                   activation=activation)
            for _ in range(stack_count)
        ])

        assert len(self.fusion_blocks) == len(self.lateral_convs)

    def upscale(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode='nearest')

    def forward(
        self,
        F5: torch.Tensor,
        S: list[torch.Tensor]
    ):
        """
        Enrich the finer-scale feature maps with semantic information, in a
        recursive manner.

        Parameters:
            F5:
                the query matrix as a feature map $(N, d_k, H, W)$
            S:
                the feature maps of the feature pyramid produced from the CNN
                backbone
        """

        F_enriched = [F5]

        num_steps = len(self.fusion_blocks)
        for step in range(num_steps):
            lateral_conv = self.lateral_convs[step]
            F_enriched[-1] = lateral_conv(F_enriched[-1])

            # Take the last feature map.
            F_coarse = F_enriched[-1]
            S_fine = S[num_steps - 1 - step]

            # Upscale the coarse query map.
            F_coarse_upscaled = self.upscale(F_coarse)

            # Imbue the semantic information to the finer feature map S[i - 1]
            # with a fusion operation.
            fuse = self.fusion_blocks[step]
            F_enriched.append(fuse(F_coarse_upscaled, S_fine))

        F_enriched.reverse()

        return F_enriched


class BottomUpFusionNet(torch.nn.Module):
    """
    Bottom-up fusion network.

    This is the convolutional network used to perform the "bottom-up
    path augmentation" as described in the paper.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stack_count: int,
        hidden_dim_expansion_factor: float = 1.0,
        repvgg_stack_depth: int = 3,
        activation: str = 'silu'
    ):
        super().__init__()
        self.downsample_convs = torch.nn.ModuleList([
            DownsampleConvolution(out_channels, out_channels,
                                  activation=activation)
            for _ in range(stack_count)
        ])
        self.fusion_blocks = torch.nn.ModuleList([
            FusionBlock(
                in_channels, out_channels,
                hidden_dim_expansion_factor=hidden_dim_expansion_factor,
                repvgg_layer_count=repvgg_stack_depth,
                activation=activation
            )
            for _ in range(stack_count)
        ])
        assert len(self.fusion_blocks) == len(self.downsample_convs)

    def forward(
        self,
        F_topdown_enriched: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Refines the feature pyramid in a recursive bottom-up manner.

        Parameters:
            F_topdown_enriched:
                the feature enriched by the top-down fusion network
        """
        # Bottom-up semantic refinement in a recursive fashion.
        F_bottomup_refined = [F_topdown_enriched[0]]

        num_steps = len(self.fusion_blocks)
        for step in range(num_steps):
            # Take the last feature map.
            F_fine = F_bottomup_refined[step]
            F_coarse = F_topdown_enriched[step + 1]

            # Downsample the fine enriched map.
            downsample = self.downsample_convs[step]
            F_fine_downsampled = downsample(F_fine)

            fuse = self.fusion_blocks[step]
            F_coarse_refined = fuse(F_fine_downsampled, F_coarse)
            F_bottomup_refined.append(F_coarse_refined)

        return F_bottomup_refined


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

    - $\mathbf{F}_3^{+} \leftarrow \mathbf{F}_3$
    - $\mathbf{F}_4^{+} \leftarrow \mathrm{refine}(\mathbf{F}_3^{+}, \mathbf{F}_4)$
    - $\mathbf{F}_5^{+} \leftarrow \mathrm{refine}(\mathbf{F}_4^{+}, \mathbf{F}_5)$

    We follow the implementation as detailed in Figure 4 of the paper.
    """

    def __init__(
        self,
        stack_count: int,
        hidden_dim: int = 256,
        hidden_dim_expansion_factor: float = 1.0,
        repvgg_stack_depth: int = 3,
        activation: str = 'silu'
    ):
        super().__init__()

        assert stack_count > 0

        self.fuse_topdown = TopDownFusionNet(
            hidden_dim, hidden_dim,
            stack_count,
            hidden_dim_expansion_factor=hidden_dim_expansion_factor,
            repvgg_stack_depth=repvgg_stack_depth,
            activation=activation
        )
        self.refine_bottomup = BottomUpFusionNet(
            hidden_dim, hidden_dim,
            stack_count,
            hidden_dim_expansion_factor=hidden_dim_expansion_factor,
            repvgg_stack_depth=repvgg_stack_depth,
            activation=activation
        )

    def forward(
        self,
        aifi_out: torch.Tensor,
        backbone_feature_pyramid: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        # Convenient aliases to retrieve ourselves with the paper.
        S = backbone_feature_pyramid
        S5 = S[-1]
        F5_flat = aifi_out

        # Reshape the object query matrix as a feature map.
        n, _, h, w = S5.shape
        _, _, c = F5_flat.shape
        F5 = F5_flat.permute(0, 2, 1).reshape(n, c, h, w)

        F_topdown_enriched = self.fuse_topdown.forward(F5, S)
        F_bottomup_refined = self.refine_bottomup.forward(F_topdown_enriched)
        return F_bottomup_refined
