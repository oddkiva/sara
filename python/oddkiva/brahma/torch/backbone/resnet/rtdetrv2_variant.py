from pathlib import Path

from loguru import logger

import torch
import torch.nn as nn
import torchvision.ops as ops

from oddkiva.brahma.torch.backbone.resnet.vanilla import (
    ConvBNA,
    make_activation_func
)


class UnbiasedConvBNA(ConvBNA):
    """This class is a convenience class that specializes the classical block
    `ConvBNA`.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, id: int, activation: str | None ='relu'):
        """Constructs an [Unbiased-Conv+BN+Activation] block.
        """
        super().__init__(in_channels, out_channels, kernel_size, stride, True,
                         activation, id, bias=False)


class ResidualBottleneckBlock(nn.Module):
    """
    This class implements a variant of fundamental residual block, which is
    used in the ResNet backbone of RT-DETR v2.

    In general, this block tweaks the original implementation to make it as
    cheap and lightweight in terms of parameters and computational cost:

    Specifically, the key differences from the classical residual bottleneck
    block are as follows.

    - Each convolution operation is unbiased.
    - The last convolution of the block does not use any activation function.
    - The downsampling of the feature map happens on the second
      `UnbiasedConvBNA` block and not on the first block.
    - The shortcut connection contains an average pooling layer if the stride
      is 2. This pooling is applied first before the convolutional operation.
    - The identity function is favored over the convolutional operation in the
      shortcut connection.

    The authors have a different vision of the residual bottleneck block and
    the differences are significant enough to justify the writing of a new
    class. This keeps the code simple.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "relu",
        make_convolutional_shortcut: bool = False,
        inplace_activation: bool = False
    ):
        """
        Constructs the residual bottleneck block variant used in RT-DETR v2.
        """

        super().__init__()
        self.convs = nn.Sequential(
            UnbiasedConvBNA(in_channels, out_channels, 1, 1,
                            0,  # Id
                            activation=activation),
            UnbiasedConvBNA(out_channels, out_channels, 3, stride,
                            1,  # Id
                            activation=activation),
            UnbiasedConvBNA(out_channels, out_channels * (2**2), 1, 1,
                            2,  # Id
                            activation=None),  # oh my! so many modifications!
        )

        if make_convolutional_shortcut:
            if stride == 1:
                self.shortcut = UnbiasedConvBNA(in_channels,
                                                out_channels * (2**2),
                                                1, 1,
                                                0, activation=None)
            elif stride == 2:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(2, 2, 0, ceil_mode=True),
                    UnbiasedConvBNA(in_channels, out_channels * (2**2),
                                    1, 1,
                                    0, activation=None)
                )
            else:
                ValueError("Unsupported: the stride must be 1 or 2!")
        else:
            self.shortcut = nn.Identity()

        # Add the activation layer
        self.activation = make_activation_func(
            activation, inplace=inplace_activation
        )

    def forward(self, x):
        out = self.convs.forward(x) + self.shortcut(x)
        if self.activation is not None:
            out = self.activation(out)

        return out

    @property
    def in_channels(self) -> int:
        return self.convs[0].layers[0].in_channels

    @property
    def out_channels(self) -> int:
        return self.convs[-1].layers[0].out_channels


class ResNet50RTDETRV2Variant(nn.Module):
    """This class implements RT-DETR v2 variant form of ResNet-50.

    This variant has some nice features that further reduces the number of
    learnable parameters and also boosts up the total FLOPs per second.

    - None of its convolutional operations contains a learnable bias.
    - In each image level, we apply a stack of residual bottleneck blocks.

      - Only the first residual bottleneck block of the stack incorporates a
        convolutional shortcut connection, contrary to the classical version of
        ResNet-50.
      - Then the next residual bottleneck blocks uses the identity shortcut
        connection, which is what the ResNet paper advocated originally.
    """

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            # Surprising feature:
            # - we downsample straight away from the first convolutional
            #   operation.
            # - this is understandable as we want to reduce computational cost
            #   as much as possible from the get-go.
            #
            # And then we downsample again with the max-pool operation.
            nn.Sequential(
                UnbiasedConvBNA(3, 32, 3, 2,   # Downsample here
                                0),            # id
                UnbiasedConvBNA(32, 32, 3, 1,
                                1),            # id
                UnbiasedConvBNA(32, 64, 3, 1,
                                1),            # id
                nn.MaxPool2d(3, stride=2, padding=1),
            ),
            # P0
            #
            # No downsampling here like in the classical ResNet-50.
            nn.Sequential(
                # Out dim is not 64! But 256 (cf. implementation).
                ResidualBottleneckBlock(64, 64, 1, "relu",
                                        make_convolutional_shortcut=True),
                ResidualBottleneckBlock(256, 64, 1, "relu"),
                ResidualBottleneckBlock(256, 64, 1, "relu"),
            ),
            # P1
            nn.Sequential(
                ResidualBottleneckBlock(256, 128, 2, "relu",
                                        make_convolutional_shortcut=True),
                ResidualBottleneckBlock(512, 128, 1, "relu"),
                ResidualBottleneckBlock(512, 128, 1, "relu"),
                ResidualBottleneckBlock(512, 128, 1, "relu"),
            ),
            # P2
            nn.Sequential(
                ResidualBottleneckBlock(512, 256, 2, "relu",
                                        make_convolutional_shortcut=True),
                ResidualBottleneckBlock(1024, 256, 1, "relu"),
                ResidualBottleneckBlock(1024, 256, 1, "relu"),
                ResidualBottleneckBlock(1024, 256, 1, "relu"),
                ResidualBottleneckBlock(1024, 256, 1, "relu"),
                ResidualBottleneckBlock(1024, 256, 1, "relu"),
            ),
            # P3
            nn.Sequential(
                ResidualBottleneckBlock(1024, 512, 2, "relu",
                                        make_convolutional_shortcut=True),
                ResidualBottleneckBlock(2048, 512, 1, "relu"),
                ResidualBottleneckBlock(2048, 512, 1, "relu"),
            )
        )

    def forward(self, x):
        return self.blocks.forward(x)

    def freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def freeze_batch_norm(self, m: nn.Module) -> nn.Module:
        if isinstance(m, nn.BatchNorm2d):
            # If m is a leaf module and that leaf module is also a BatchNorm2d
            # module.
            m = ops.FrozenBatchNorm2d(m.num_features)
        else:
            # DFS visit.
            for child_tree_name, child_tree in m.named_children():
                # Go to the child trees.
                child_tree_transmuted = self.freeze_batch_norm(child_tree)

                # If the child tree has transmuted to a new child object.
                #
                # A child tree transmutes if we create a new object referenced
                # by a new "pointer" value.
                #
                # In practice we only leaf nodes tha are BatchNorm2d operations
                # and it has no children. So this copy-pasted code is a bit
                # strange.
                if child_tree_transmuted is not child_tree:
                    logger.debug(
                        f"child_tree has transmuted from {child_tree} to {child_tree_transmuted}")
                    # Update the child.
                    setattr(m, child_tree_name, child_tree_transmuted)
                # else:
                #     logger.debug(
                #         f"child_tree has not transmuted: {child_tree_transmuted}"
                #     )
        return m

    @property
    def bottleneck_stacks(self):
        return self.blocks[1:]

    @property
    def feature_pyramid_dims(self, include_first_convolutional_stack: bool = False):
        dims = [
            bottleneck_stack[-1].out_channels
            for bottleneck_stack in self.bottleneck_stacks
        ]
        if include_first_convolutional_stack:
            dim = self.blocks[0][-2].layers[0].out_channels
            dims = [dim] + dims
        return dims
