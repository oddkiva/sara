# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from collections import OrderedDict

from loguru import logger

import torch.nn as nn
import torchvision.ops as ops


def make_activation_func(
    activation: str | None,
    inplace: bool = False
) -> nn.Module | None:
    """Make an activation function.

    parameters:
        activation:
            options are `leaky`, `relu`, `mish`, `linear`, `logistic`,
            `silu`, `gelu`, `None`.
    """
    # Add the activation layer
    if activation == "leaky":
        return nn.LeakyReLU(0.1, inplace=inplace)
    if activation == "relu":
        return nn.ReLU(inplace=inplace)
    if activation == "mish":
        return nn.Mish(inplace=inplace)
    if activation == "linear":
        return nn.Identity(inplace=inplace)
    if activation == "logistic":
        return nn.Sigmoid()
    if activation == "silu":
        return nn.SiLU(inplace=inplace)
    if activation == "gelu":
        return nn.GELU()

    if activation is None:
        return None

    logger.error(f"No convolutional activation named {activation}")
    return None


class ConvBNA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        batch_normalize: bool,
        activation: str | None,
        id: int | None = None,
        bias: bool = True,
        inplace_activation: bool = True,
        freeze_batch_norm: bool = False
    ):
        super(ConvBNA, self).__init__()
        self.layers = nn.Sequential()

        pad_size = (kernel_size - 1) // 2

        # Add the convolutional layer
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=pad_size,
            bias=bias,
            padding_mode="zeros",  # Let's be explicit about the padding value
        )
        conv_name = 'conv'
        if id is not None:
            conv_name = f'{conv_name}_{id}'
        self.layers.add_module(conv_name, conv)

        if batch_normalize:
            batch_norm_name = 'batch_norm'
            if id is not None:
                batch_norm_name = f'{batch_norm_name}_{id}'

            batch_norm_layer = \
                ops.FrozenBatchNorm2d(out_channels) if freeze_batch_norm else \
                nn.BatchNorm2d(out_channels)
            self.layers.add_module(batch_norm_name, batch_norm_layer)

        activation_fn = make_activation_func(
            activation, inplace=inplace_activation
        )

        if activation_fn is not None:
            activation_name = 'activation'
            if id is not None:
                activation_name = f'{activation_name}_{id}'
            self.layers.add_module(activation_name, activation_fn)

    def forward(self, x):
        return self.layers.forward(x)


class ResidualBottleneckBlock(nn.Module):
    """
    This class implements the fundamental residual block used in the classical
    ResNet architecture.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "relu",
        bias: bool = True,
        batch_normalize_after_shortcut: bool = False
    ):
        """Constructs a Residual Bottleneck Block used in ResNet.

        Parameters:
            in_channels:
                the input feature dimension.
            out_channels:
                the output feature dimension.
            stride:
                the spatial step size used for downsampling purposes.
            activation:
                the activation function.
            bias:
                add learnable biases for every convolutional operations.
            batch_normalize_after_shortcut:
                optionally add a post-batch-normalization layer to the shortcut
                branch.

        Notes
        -----
        Unless you have a particular need like in RT-DETR v2, you should leave
        the default parameter:

        - `use_shortcut_connection` value to `True`
        - `batch_normalize_after_shortcut` value to `False`
        """
        super().__init__()
        self.convs = nn.Sequential(
            ConvBNA(in_channels, out_channels, 1, stride,
                    True,        # Batch-normalization
                    activation,  # Activation
                    id=0,        # Id
                    bias=bias),
            ConvBNA(out_channels, out_channels, 3, 1,
                    True,        # Batch-normalization
                    activation,  # Activation
                    id=1,        # Id
                    bias=bias),
            ConvBNA(out_channels, out_channels * (2**2), 1, 1,
                    True,        # Batch-normalization
                    activation,  # Activation
                    id=2,        # Id
                    bias=bias),
        )

        self.shortcut = ConvBNA(
            in_channels, out_channels * (2**2), 1, stride,
            batch_normalize_after_shortcut,
            "linear",
            bias=bias
        )

        # Add the activation layer
        self.activation = make_activation_func(activation)

        self._in_channels = in_channels
        self._out_channels = out_channels

    def forward(self, x):
        assert self.activation is not None
        if self.shortcut is None:
            return self.activation(self.convs(x))
        return self.activation(self.convs.forward(x) + self.shortcut(x))

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels


class ResNet50(nn.Module):
    """
    The ResNet-50 architecture.
    """

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            # From the original paper.
            ConvBNA(3, 64, 7, 2, True, "relu", 0),
            nn.AvgPool2d(3, 2),
            # P0
            nn.Sequential(
                # Reminder: the output dim is misleading and is not 64!
                # But 256 = 64 x 2^2 (cf. implementation).
                ResidualBottleneckBlock(64, 64, 1, "relu"),
                ResidualBottleneckBlock(256, 64, 1, "relu"),
                ResidualBottleneckBlock(256, 64, 1, "relu"),
            ),
            # P1
            nn.Sequential(
                ResidualBottleneckBlock(256, 128, 2, "relu"),
                ResidualBottleneckBlock(512, 128, 1, "relu"),
                ResidualBottleneckBlock(512, 128, 1, "relu"),
                ResidualBottleneckBlock(512, 128, 1, "relu"),
            ),
            # P2
            nn.Sequential(
                ResidualBottleneckBlock(512, 256, 2, "relu"),
                ResidualBottleneckBlock(1024, 256, 1, "relu"),
                ResidualBottleneckBlock(1024, 256, 1, "relu"),
                ResidualBottleneckBlock(1024, 256, 1, "relu"),
                ResidualBottleneckBlock(1024, 256, 1, "relu"),
                ResidualBottleneckBlock(1024, 256, 1, "relu"),
            ),
            # P3
            nn.Sequential(
                ResidualBottleneckBlock(1024, 512, 2, "relu"),
                ResidualBottleneckBlock(2048, 512, 1, "relu"),
                ResidualBottleneckBlock(2048, 512, 1, "relu"),
            ),
        )

    def forward(self, x):
        return self.blocks.forward(x)


class ResNet50Variant(nn.Module):
    """This class implements an experimental ResNet-50 variant where we replace
    the first 7x7 convolution with a sequence of residual bottleneck blocks.
    """

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            # I don't like the 7x7 convolution:
            # ConvBNA(3, 64, 7, 2, True, "relu", 0),
            # nn.AvgPool2d(3, 2),
            #
            # Instead I will use 3 convolutions, each of one them having kernel
            # of 3x3 filters. Applying those 3 3x3 convolutions successively is
            # equivalent to a 7x7 convolution.
            nn.Sequential(
                OrderedDict([
                    # The composition is the 3 residual blocks has a receptive
                    # field of 7x7
                    # x |-> conv1x1 conv3x3 conv1x1)(x) + shortcut(x)
                    ("conv3x3_step0", ResidualBottleneckBlock(3, 16, 1, "relu")),
                    # x |-> conv1x1 conv3x3 conv1x1)(x) + shortcut(x)
                    ("conv3x3_step1", ResidualBottleneckBlock(64, 16, 1, "relu")),
                    # x |-> conv1x1 conv3x3 conv1x1)(x) + shortcut(x)
                    ("conv3x3_step2", ResidualBottleneckBlock(64, 16, 2, "relu")),

                    # This will replace nn.AvgPool2d
                    ("avg_pool_2d_alternative", ResidualBottleneckBlock(64, 16, 2, "relu")),
                ])
            ),
            # P0
            nn.Sequential(
                # Out dim is not 64! But 256 (cf. implementation).
                ResidualBottleneckBlock(64, 64, 1, "relu"),
                ResidualBottleneckBlock(256, 64, 1, "relu"),
                ResidualBottleneckBlock(256, 64, 1, "relu"),
            ),
            # P1
            nn.Sequential(
                ResidualBottleneckBlock(256, 128, 2, "relu"),
                ResidualBottleneckBlock(512, 128, 1, "relu"),
                ResidualBottleneckBlock(512, 128, 1, "relu"),
                ResidualBottleneckBlock(512, 128, 1, "relu"),
            ),
            # P2
            nn.Sequential(
                ResidualBottleneckBlock(512, 256, 2, "relu"),
                ResidualBottleneckBlock(1024, 256, 1, "relu"),
                ResidualBottleneckBlock(1024, 256, 1, "relu"),
                ResidualBottleneckBlock(1024, 256, 1, "relu"),
                ResidualBottleneckBlock(1024, 256, 1, "relu"),
                ResidualBottleneckBlock(1024, 256, 1, "relu"),
            ),
            # P3
            nn.Sequential(
                ResidualBottleneckBlock(1024, 512, 2, "relu"),
                ResidualBottleneckBlock(2048, 512, 1, "relu"),
                ResidualBottleneckBlock(2048, 512, 1, "relu"),
            ),
        )

    def forward(self, x):
        return self.blocks.forward(x)
