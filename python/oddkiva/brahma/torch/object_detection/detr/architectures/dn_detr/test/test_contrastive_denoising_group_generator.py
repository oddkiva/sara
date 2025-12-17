# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import pickle
from loguru import logger

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader

import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.datasets.coco.dataloader import collate_fn
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    dn_detr.contrastive_denoising_group_generator import (
        ContrastiveDenoisingGroupGenerator
    )


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


def test_contrastive_denoising_group_generator():
    dng = ContrastiveDenoisingGroupGenerator(80)

    images, boxes, labels = get_or_create_coco_batch_sample()

    assert images.shape == (16, 3, 640, 640)
    assert len(boxes) == 16
    assert len(labels) == 16

    torch.manual_seed(0)
    g = dng.forward(300, boxes, labels)
