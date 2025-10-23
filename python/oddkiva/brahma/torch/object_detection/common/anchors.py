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
        x_axis = torch.arange(0, w)
        y_axis = torch.arange(0, h)

        x, y = torch.meshgrid(x_axis, y_axis, indexing='xy')
        xy = torch.stack((x, y), dim=-1)\
            .to(dtype=torch.float32)\
            .reshape((num_boxes, 2))

        # The box centers are the centers of each pixel of the image
        box_centers = xy + 0.5

        # Form the tensor of box sizes.
        box_sizes_tensorized = torch.tensor(box_sizes)\
            .to(dtype=torch.float32)\
            .unsqueeze(0)\
            .repeat((num_boxes, 1))

        # Stack the box centers and box sizes
        boxes = torch.cat((box_centers, box_sizes_tensorized), dim=-1)

        if normalize_geometry:
            boxes[:, (0, 2)] /= w
            boxes[:, (1, 3)] /= h

    return boxes


def rescale_boxes():
     pass
