# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from collections import OrderedDict

import torch.nn as nn


def make_activation_func(activation: str | None, inplace: bool = False):
    # Add the activation layer
    if activation == "leaky":
        return nn.LeakyReLU(0.1, inplace=inplace)
    elif activation == "relu":
        return nn.ReLU(inplace=inplace)
    elif activation == "mish":
        return nn.Mish(inplace=inplace)
    elif activation == "linear":
        return nn.Identity(inplace=inplace)
    elif activation == "logistic":
        return nn.Sigmoid()
    if activation == "silu":
        return nn.SiLU(inplace=inplace)
    elif activation is None:
        return None
    else:
        raise ValueError(f"No convolutional activation named {activation}")


class ConvBNA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        batch_normalize: bool,
        activation: str | None,
        id: int,
        inplace_activation: bool = False
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
            bias=True,
            padding_mode="zeros",  # Let's be explicit about the padding value
        )
        self.layers.add_module(f"conv_{id}", conv)
        if batch_normalize:
            self.layers.add_module(
                f"batch_norm_{id}", nn.BatchNorm2d(out_channels)
            )

        activation_fn = make_activation_func(activation,
                                             inplace=inplace_activation)
        if activation_fn is not None:
            self.layers.add_module(f"{activation}_{id}", activation_fn)

    def forward(self, x):
        return self.layers.forward(x)


class ResidualBottleneckBlock(nn.Module):
    """
    This class implements the fundamental residual block used in ResNet.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "relu",
    ):
        super().__init__()
        self.convs = nn.Sequential(
            ConvBNA(in_channels, out_channels, 1, stride, True, activation, 0),
            ConvBNA(out_channels, out_channels, 3, 1, True, activation, 1),
            ConvBNA(
                out_channels,
                out_channels * (2**2),
                1,
                1,
                True,
                activation,
                2,
            ),
        )

        self.shortcut = ConvBNA(
            in_channels, out_channels * (2**2), 1, stride, False, "linear", 0
        )

        # Add the activation layer
        if activation == "leaky":
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "mish":
            self.activation = nn.Mish()
        elif activation == "linear":
            self.activation = nn.Identity(inplace=True)
        elif activation == "logistic":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"No convolutional activation named {activation}")

        self._in_channels = in_channels
        self._out_channels = out_channels

    def forward(self, x):
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
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            # I don't like the 7x7 convolution:
            # ConvBNA(3, 64, 7, 2, True, "relu", 0),
            # nn.AvgPool2d(3, 2),
            #
            # Instead I will use 3x3 filters:
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
