import random

import torch
import torch.nn.functional as F


def collate_fn(
    data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    # pack the images together into a single tensor of shape (n, c, h, w).
    images = torch.stack([sample[0] for sample in data], dim=0)

    # pack the annotated box data into a list of tensors.
    boxes = [sample[1] for sample in data]
    # likewise for the labels.
    labels = [sample[2] for sample in data]

    return images, boxes, labels


class RTDETRImageCollateFunction:
    """ Resample the batch of images at different sizes.
    """

    # Taken from the original implementation.
    DEFAULT_RESAMPLING_SIZES = [
        480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800
    ]

    def __init__(
        self,
        resampling_sizes: list[int | tuple[int, int]] | None = None
    ):
        if resampling_sizes is None:
            self.resampling_sizes = self.DEFAULT_RESAMPLING_SIZES
        else:
            self.resampling_sizes = resampling_sizes

    def __call__(self, data):
        # pack the images together into a single tensor of shape (N, C, H, W).
        images = torch.stack([sample[0] for sample in data], dim=0)
        im_szs = random.choice(self.resampling_sizes)
        if type(im_szs) is int:
            im_szs = (im_szs, im_szs)
        images = F.interpolate(images, size=im_szs)

        # pack the annotated box data into a list of tensors.
        boxes = [sample[1] for sample in data]
        for b in boxes:
            b.canvas_size = im_szs

        # likewise for the labels.
        labels = [sample[2] for sample in data]

        return images, boxes, labels
