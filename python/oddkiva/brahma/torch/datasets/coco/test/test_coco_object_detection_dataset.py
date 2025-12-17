# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from oddkiva.brahma.torch.datasets.coco import COCOObjectDetectionDataset
from torchvision.transforms import v2


def test_coco_dataset():
    coco = COCOObjectDetectionDataset(
        train_or_val='train',
        transform=v2.Compose([
        ])
    )

    def collate_fn(
        data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ):
        # Pack the images together into a single tensor of shape (N, C, H, W).
        images = torch.stack([sample[0] for sample in data], dim=0)
        # Pack the annotated box data into a list of tensors.
        boxes = [sample[1] for sample in data]
        # Likewise for the labels.
        labels = [sample[2] for sample in data]
        annotations = {
            'boxes': boxes,
            'labels': labels
        }
        return (images, annotations)

    coco_dl = DataLoader(
        dataset=coco_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )
    img, boxes, labels = iter(coco_dl)
    assert img.shape == (16, 3, 640, 640)
    assert len(boxes) == 16
    assert len(labels) == 16
