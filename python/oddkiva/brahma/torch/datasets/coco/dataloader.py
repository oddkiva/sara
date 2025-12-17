import torch


def collate_fn(
    data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    # pack the images together into a single tensor of shape (n, c, h, w).
    images = torch.stack([sample[0] for sample in data], dim=0)
    sz = random.choice(self.RTDETR_TRAIN_IMAGE_SQUARE_SIZES)
    images = F.interpolate(images, size=sz)

    # pack the annotated box data into a list of tensors.
    boxes = [sample[1] for sample in data]
    # likewise for the labels.
    labels = [sample[2] for sample in data]
    annotations = {
        'boxes': boxes,
        'labels': labels
    }

    return labels, annotations


class RTDETRImageCollateFunction:

    RTDETR_TRAIN_IMAGE_SQUARE_SIZES = [
        480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800
    ]

    def __call__(self, data):
        # pack the images together into a single tensor of shape (n, c, h, w).
        images = torch.stack([sample[0] for sample in data], dim=0)
        sz = random.choice(self.RTDETR_TRAIN_IMAGE_SQUARE_SIZES)
        images = F.interpolate(images, size=sz)

        # pack the annotated box data into a list of tensors.
        boxes = [sample[1] for sample in data]
        # likewise for the labels.
        labels = [sample[2] for sample in data]
        annotations = {
            'boxes': boxes,
            'labels': labels
        }
        return (images, annotations)
