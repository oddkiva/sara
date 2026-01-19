# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from loguru import logger

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader

import oddkiva.sara as sara
import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva.brahma.torch.datasets.coco.dataloader import collate_fn
from oddkiva.brahma.torch.object_detection.common.data_transforms import (
    ToNormalizedCXCYWHBoxes
)

def get_coco_batch_sample():
    logger.info(f"Instantiating COCO dataset...")
    transform = v2.Compose([
        v2.RandomIoUCrop(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Resize((640, 640)),
        v2.SanitizeBoundingBoxes()
    ])
    coco_ds = coco.COCOObjectDetectionDataset(
        train_or_val='val',
        transform=transform
    )
    logger.info(f"Number of samples: {len(coco_ds)}")

    logger.info(f"Instantiating COCO dataloader...")
    coco_dl = DataLoader(
        dataset=coco_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )

    logger.info(f"Getting first batch sample from COCO dataloader...")
    coco_it = iter(coco_dl)
    sample = next(coco_it)

    return sample


def test_box_normalization():
    with sara.Timer("Get Sample"):
        sample = get_coco_batch_sample()

    normalize_boxes = ToNormalizedCXCYWHBoxes()

    img, boxes, labels = sample
    img_n, boxes_n, labels_n = normalize_boxes(sample)

    # Check that the image data is unchanged.
    assert torch.equal(img, img_n)

    # Check that the box coordinate normalization.
    for b_i, bn_i in zip(boxes, boxes_n):
        l, t, w, h = b_i.unbind(-1)
        cx = l + 0.5 * w
        cy = t + 0.5 * h
        cxcywh = torch.stack((cx, cy, w, h), dim=-1)
        whwh = torch.tensor(b_i.canvas_size[::-1]).tile(2)
        assert torch.dist(cxcywh / whwh, bn_i) < 1e-6

    # Check that the label data is unchanged.
    for l, ln in zip(labels, labels_n):
        assert torch.equal(l, ln)
