import torch

from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.encoder.ccff import (
        DownsampleConvolution,
        LateralConvolution
    )


def test_lateral_convolution():
    in_channels_sequence = [512, 1024, 2048]
    hidden_dim = 256
    for in_channels in in_channels_sequence:
        block = LateralConvolution(in_channels, hidden_dim)

        assert len(block.layers) == 2

        assert type(block.layers[0]) is torch.nn.Conv2d
        assert block.layers[0].in_channels == in_channels
        assert block.layers[0].out_channels == hidden_dim
        assert block.layers[0].kernel_size == (1, 1)
        assert block.layers[0].stride == (1, 1)

        assert type(block.layers[1]) is torch.nn.BatchNorm2d
        assert block.layers[1].num_features == hidden_dim


def test_downsample_convolution():
    in_channels_sequence = [512, 1024, 2048]
    hidden_dim = 256
    for (id, in_channels) in enumerate(in_channels_sequence):
        block = DownsampleConvolution(in_channels, hidden_dim, id,
                                      activation='silu')

        assert len(block.layers) == 3
        assert type(block.layers[0]) is torch.nn.Conv2d
        assert block.layers[0].in_channels == in_channels
        assert block.layers[0].out_channels == hidden_dim
        assert block.layers[0].kernel_size == (3, 3)
        assert block.layers[0].stride == (2, 2)

        assert type(block.layers[1]) is torch.nn.BatchNorm2d
        assert block.layers[1].num_features == hidden_dim

        assert type(block.layers[2]) is torch.nn.SiLU
