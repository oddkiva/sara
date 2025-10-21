import torch


def enumerate_boxes(image_shape: tuple[int, int],
                    box_sizes: list[tuple[int, int]]) -> torch.Tensor:
    h, w = image_shape
    x_axis = torch.arange(0, w)
    y_axis = torch.arange(0, h)

    # Box centers.
    x, y = torch.meshgrid(x_axis, y_axis, indexing='xy')
    box_centers = torch.stack((x, y))

    num_boxes = box_centers.shape[0]

    boxes = []

    for sz in box_sizes:
        sz_tensorized = torch.tensor(sz).repeat(num_boxes)
        boxes_with_sz = torch.cat((box_centers, sz_tensorized))
        boxes.append(boxes_with_sz)

    return torch.cat(boxes, dim=-1)


def rescale_boxes():
     pass
