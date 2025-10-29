# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torch

import oddkiva.brahma.torch.image_processing.warp as W


def test_enumerate_coords():
    coords = W.enumerate_coords(3, 4)
    assert torch.equal(
        coords,
        torch.Tensor(
            [
                [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            ]
        ),
    )


def test_filter_coords():
    coords = torch.Tensor([[0, 1], [1, 2]])

    w, h = 3, 4
    x = torch.zeros((h, w))

    ixs = (coords[1, :] * w + coords[0, :]).int()
    x.flatten()[ixs] = 1

    assert torch.equal(
        x,
        # fmt: off
        torch.Tensor([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
    )
    # fmt: on


def test_bilinear_interpolation_2d():
    # fmt: off
    values = torch.Tensor([[0., 1.],
                           [2., 3.],
                           [4., 5.]])
    # fmt: on

    x = torch.Tensor([0.5, 0.5, 0.5])
    y = torch.Tensor([0.5, 1.5, 1.5])
    coords = torch.stack((x, y))

    interp_values, _ = W.bilinear_interpolation_2d(values, coords)
    assert torch.equal(interp_values, torch.Tensor([1.5, 3.5, 3.5]))
