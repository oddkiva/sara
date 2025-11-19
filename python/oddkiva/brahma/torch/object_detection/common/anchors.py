# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torch


def enumerate_anchors(image_sizes: tuple[int, int],
                      box_sizes: tuple[int, int],
                      normalize_geometry: bool,
                      device: torch.device) -> torch.Tensor:
    """Enumerate all the anchors

    For a given feature map X, an object detector will typically predict the
    geometry of the object box in each pixel (x, y) of the the feature map,
    under the hypothesis that the pixel (x, y) contains an object.

    The geometry of object box is initialized from a fairly good heuristic:
    - the centre of the box is the pixel center (x + 0.5, y + 0.5).
    - the box can be initialized as a rectangle of side length equal to
      5%, 10%, 20% of the actual image sizes.

    That box geometry initialization is called an anchor and is encoded as the
    4D vector (xc, yc, w, h)
    where:
    - (xc, yc) is the center of the box
    - (w, h) is the sizes of the box
    """

    w, h = image_sizes
    num_boxes = h * w

    # Box centers.
    with device:
        x_axis = torch.arange(w, dtype=torch.float32)
        y_axis = torch.arange(h, dtype=torch.float32)

        x, y = torch.meshgrid(x_axis, y_axis, indexing='xy')
        xy = torch.stack((x, y), dim=-1)\
            .reshape((num_boxes, 2))

        # The box centers are the centers of each pixel of the image
        box_centers = xy + 0.5

        # Form the tensor of box sizes.
        box_sizes_tensorized = torch.tensor(box_sizes)\
            .to(dtype=torch.float32)\
            .unsqueeze(0)\
            .repeat((num_boxes, 1))

        if normalize_geometry:
            # This is faster but a lot less precise.
            wh_inverse = torch.tensor([1. / w, 1. / h])
            box_centers = box_centers * wh_inverse
            box_sizes_tensorized = box_sizes_tensorized * wh_inverse
            # Decide what we want to do.
            # wh = torch.tensor([w, h], dtype=torch.float32)
            # box_centers = box_centers / wh
            # box_sizes_tensorized = box_sizes_tensorized / wh

        # Stack the box centers and box sizes
        boxes = torch.cat((box_centers, box_sizes_tensorized), dim=-1)

    return boxes


def enumerate_anchor_pyramid(
    pyramid_image_sizes: list[tuple[int, int]],
    normalized_base_box_size: float = 0.05,
    normalize_anchor_geometry: bool = True,
    device: torch.device = torch.device('cpu')
) -> list[torch.Tensor]:
    level_count = len(pyramid_image_sizes)

    box_sizes_normalized = [
        # 2x bigger boxes for coarser and coarser image levels.
        normalized_base_box_size * (2 ** lvl)
        for lvl in range(level_count)
    ]
    box_sizes_per_level = [
        (f * w, f * h)
        for f, (w, h) in zip(box_sizes_normalized, pyramid_image_sizes)
    ]

    anchors = [
        enumerate_anchors(wh, box_sizes, normalize_anchor_geometry, device)
        for wh, box_sizes in zip(pyramid_image_sizes, box_sizes_per_level)
    ]

    return anchors


def calculate_anchor_logits(
    anchors: torch.Tensor,
    eps: float = 1e-2
) -> tuple[torch.Tensor, torch.Tensor]:
    # First filter out anchors that are ill-defined for the inverse sigmoid
    # function.
    anchor_valid_mask = ((anchors > eps) * (anchors < (1 - eps)))\
        .all(-1, keepdim=True)

    # The sigmoid function being an increasing activation function, we can
    # calculate explicitly the logits which are the inverse of the sigmoid
    # values.
    anchor_logits = torch.log(anchors / (1 - anchors))
    anchor_logits = torch.where(anchor_valid_mask, anchor_logits, torch.inf)

    return anchor_logits, anchor_valid_mask
