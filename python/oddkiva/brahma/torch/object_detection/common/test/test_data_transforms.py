import pickle

from loguru import logger

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.object_detection.common.data_transforms import (
    ToNormalizedCXCYWHBoxes
)
import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.datasets.coco.dataloader import collate_fn


def get_or_create_coco_batch_sample(force_recreate: bool = False):
    COCO_BATCH_SAMPLE = DATA_DIR_PATH / 'coco_batch_sample.pkl'
    if COCO_BATCH_SAMPLE.exists() and not force_recreate:
        logger.info(f"Loading COCO batch sample from {COCO_BATCH_SAMPLE}...")
        with open(COCO_BATCH_SAMPLE, 'rb') as f:
            sample = pickle.load(f)
    else:
        logger.info(f"Instantiating COCO dataset...")
        transform = v2.Compose([
            v2.RandomIoUCrop(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Resize((640, 640)),
            v2.SanitizeBoundingBoxes()
        ])
        coco_ds = coco.COCOObjectDetectionDataset(
            train_or_val='train',
            transform=transform
        )

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

        with open(COCO_BATCH_SAMPLE, 'wb') as f:
            pickle.dump(sample, f)

    return sample


def test_box_normalization():
    sample = get_or_create_coco_batch_sample()

    normalize_boxes = ToNormalizedCXCYWHBoxes()

    sample_normalized = normalize_boxes(sample)
    img, boxes, labels = sample
    img_n, boxes_n, labels_n = sample_normalized

    # Check that the image data is unchanged.
    assert torch.equal(img, img_n)

    # Check that the box coordinate normalization.
    for b, bn in zip(boxes, boxes_n):
        l, t, w, h = b.unbind(-1)
        cx = l + 0.5 * w
        cy = t + 0.5 * h
        cxcywh = torch.stack((cx, cy, w, h), dim=-1)
        whwh = torch.tensor(b.canvas_size[::-1]).tile(2)

        assert torch.dist(cxcywh / whwh, bn) < 1e-6

    # Check that the label data is unchanged.
    for l, ln in zip(labels, labels_n):
        assert torch.equal(l, ln)
