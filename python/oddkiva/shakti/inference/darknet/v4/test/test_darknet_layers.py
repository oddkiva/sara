import numpy as np

import torch

from oddkiva.shakti.inference.darknet.torch_layers import (
    MaxPool,
    ConvBNA,
)


def test_mish():
    x_np = np.arange(9).reshape(1, 1, 3, 3).astype(np.float32)
    x = torch.tensor(x_np)

    # Darknet layer
    mish = torch.nn.Mish()
    y = mish(x).numpy()

    # Ground-truth
    y_true = x_np * np.tanh(np.log(1 + np.exp(x_np)))

    assert np.linalg.norm(y - y_true) < 1e-6

def test_maxpool():
    for sz in range(1, 10):
        print(f'\nsz = {sz}')
        w, h = sz, sz
        x_np = np.arange(sz ** 2).reshape(1, 1, h, w).astype(np.float32)
        x = torch.tensor(x_np)
        print(x)

        max_pool = MaxPool(2, 2)

        y = max_pool(x)
        print(y)

        # Just check the dimensions for now.
        hy, wy = y.shape[2:]
        assert wy == (sz + 1) // 2 and hy == (sz + 1) // 2

def test_convolution():
    in_channels = 3

    params = {
        'batch_normalize': True,
        'filters': 32,
        'size': 3,
        'stride': 1,
        'pad': 1,
        'activation': 'mish'
    }

    x = np.zeros((1, 3, 3, 3), dtype=np.float32)
    for c in range(x.shape[1]):
        x[0, c, :, :] = np.arange(9).reshape((3, 3)).astype(np.float32)
    x = torch.tensor(x)
    conv_bn_mish = ConvBNA(in_channels, params, id = 0)

    y = conv_bn_mish(x)

    assert x.shape == (1, 3, 3, 3)
    assert y.shape == (1, 32, 3, 3)

