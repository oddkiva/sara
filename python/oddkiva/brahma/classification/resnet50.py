import torch.nn as nn


class ConvBNA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        batch_normalize: bool,
        activation: str,
        id: int,
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

        # Add the activation layer
        if activation == "leaky":
            activation_fn = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "relu":
            activation_fn = nn.ReLU(inplace=True)
        elif activation == "mish":
            activation_fn = nn.Mish()
        elif activation == "linear":
            activation_fn = nn.Identity(inplace=True)
        elif activation == "logistic":
            activation_fn = nn.Sigmoid()
        else:
            raise ValueError(f"No convolutional activation named {activation}")
        self.layers.add_module(f"{activation}_{id}", activation_fn)

    def forward(self, x):
        return self.layers.forward(x)


class ResidualBottleneckBlock(nn.Module):
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

    def forward(self, x):
        return self.activation(self.convs.forward(x) + self.shortcut(x))


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBNA(3, 64, 7, 2, True, "relu", 0),
            nn.AvgPool2d(3, 2),
            # P0
            nn.Sequential(
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
