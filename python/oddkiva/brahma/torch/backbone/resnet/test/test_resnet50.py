# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import numpy as np

import torch

import oddkiva.brahma.torch.backbone.resnet.vanilla as R
from oddkiva.brahma.torch import DEFAULT_DEVICE


def test_conv_bn_activation_block():
    x_np = np.arange(9).reshape(1, 1, 3, 3).astype(np.float32)
    x = torch.tensor(x_np, device=DEFAULT_DEVICE)

    conv_bn_act = R.ConvBNA(1, 64, 3, 1, True, "relu", 0).to(DEFAULT_DEVICE)
    y = conv_bn_act.forward(x)

    assert y.shape == (1, 64, 3, 3)


def test_residual_bottleneck_block():
    x_np = np.arange(9).reshape(1, 1, 3, 3).astype(np.float32)
    x = torch.tensor(x_np, device=DEFAULT_DEVICE)

    block = R.ResidualBottleneckBlock(1, 8, 2).to(DEFAULT_DEVICE)
    y = block.forward(x)

    assert y.shape == (1, 32, 2, 2)


def test_resnet50():
    x_np = np.zeros((1, 3, 256, 256)).astype(np.float32)
    x = torch.tensor(x_np, device=DEFAULT_DEVICE)

    resnet50 = R.ResNet50().to(DEFAULT_DEVICE)
    y = resnet50.forward(x)

    assert y.shape == (1, 2048, 8, 8)


def test_resnet50_variant():
    x_np = np.zeros((1, 3, 256, 256)).astype(np.float32)
    x = torch.tensor(x_np, device=DEFAULT_DEVICE)

    resnet50 = R.ResNet50Variant().to(DEFAULT_DEVICE)
    y = resnet50.forward(x)

    assert y.shape == (1, 2048, 8, 8)
