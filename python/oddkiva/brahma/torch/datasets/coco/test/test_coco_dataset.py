# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from torch.utils.data import DataLoader
from torchvision.transforms import v2

from oddkiva.brahma.torch.datasets.coco import COCOObjectDetectionDataset
from oddkiva.brahma.torch.datasets.coco.dataloader import collate_fn


def test_coco_dataset():
    coco_ds = COCOObjectDetectionDataset(
        train_or_val='train',
        transform=v2.Compose([
        ])
    )

    coco_dl = DataLoader(
        dataset=coco_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )
    coco_it = iter(coco_dl)
    img, boxes, labels = next(coco_it)
    assert img.shape == (16, 3, 640, 640)
    assert len(boxes) == 16
    assert len(labels) == 16
