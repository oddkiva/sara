import torch

from oddkiva.brahma.torch import DEFAULT_DEVICE
from oddkiva.brahma.torch.object_detection.common.anchors import (
    enumerate_anchors
)


def test_enumerate_boxes():
    w = 3
    h = 2
    image_sizes = (w, h)
    box_sizes = (1, 1)

    device = torch.device(DEFAULT_DEVICE, 0)
    anchors_unnormalized = enumerate_anchors(image_sizes, box_sizes, False, device)
    anchors_normalized = enumerate_anchors(image_sizes, box_sizes, True, device)

    anchors_unnormalized_true = torch.tensor([
        [0.5, 0.5, 1., 1.],
        [1.5, 0.5, 1., 1.],
        [2.5, 0.5, 1., 1.],
        [0.5, 1.5, 1., 1.],
        [1.5, 1.5, 1., 1.],
        [2.5, 1.5, 1., 1.],
    ], device=device)

    anchors_normalized_true = torch.tensor([
        [0.5 / 3., 0.5 / 2., 1. / 3., 1. / 2.],
        [1.5 / 3., 0.5 / 2., 1. / 3., 1. / 2.],
        [2.5 / 3., 0.5 / 2., 1. / 3., 1. / 2.],
        [0.5 / 3., 1.5 / 2., 1. / 3., 1. / 2.],
        [1.5 / 3., 1.5 / 2., 1. / 3., 1. / 2.],
        [2.5 / 3., 1.5 / 2., 1. / 3., 1. / 2.],
    ], device=device)

    assert torch.norm(anchors_unnormalized - anchors_unnormalized_true) < 1e-6
    assert torch.norm(anchors_normalized - anchors_normalized_true) < 1e-6
