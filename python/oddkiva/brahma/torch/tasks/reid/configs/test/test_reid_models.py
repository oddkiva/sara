# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import numpy as np

import torch

import oddkiva.brahma.torch.tasks.reid.configs.reid_resnet as R
from oddkiva.brahma.torch import DEFAULT_DEVICE


def test_resnet50variant_reid():
    x_np = np.zeros((1, 3, 256, 256)).astype(np.float32)
    x = torch.tensor(x_np, device=DEFAULT_DEVICE)

    resnet50 = R.ResNet50VariantReid().to(DEFAULT_DEVICE)
    y = resnet50.forward(x)

    assert y.shape == (1, 256)
