# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import pickle
from collections.abc import Iterable
from loguru import logger

import matplotlib.pyplot as plt
import numpy as np

from PySide6.QtGui import QFont, QFontMetrics

import torchvision.transforms.v2 as v2
from torch.types import Number

import oddkiva.sara as sara
import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    dn_detr.query_denoiser import ContrastiveDenoisingGroupGenerator


def get_or_create_coco_dataset(force_recreate: bool = False) -> coco.COCOObjectDetectionDataset:
    coco_fp = DATA_DIR_PATH / 'coco_object_detection_dataset.pkl'
    if force_recreate:
        logger.info("Reserializing the COCO object detection dataset...")
    if coco_fp.exists() and not force_recreate:
        logger.info( f'Loading COCO Dataset object from file: {coco_fp}...')
        with open(coco_fp, 'rb') as f:
            coco_ds = pickle.load(f)
            assert type(coco_ds) is coco.COCOObjectDetectionDataset
    else:
        with sara.Timer("COCO Dataset Generation"):
            logger.info( f'Generating COCO Dataset object from file: {coco_fp}')
            transform = v2.Compose([
                v2.RandomIoUCrop(),
                v2.RandomHorizontalFlip(p=0.5),
                # v2.ToDtype(torch.float32, scale=True),
                # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                v2.SanitizeBoundingBoxes()
            ])
            coco_ds = coco.COCOObjectDetectionDataset(
                train_or_val='train',
                transform=transform
            )
        with sara.Timer("COCO Dataset Serialization"):
            logger.info( f'Serializing COCO Dataset object to file: {coco_fp}...')
            with open(coco_fp, 'ab') as f:
                pickle.dump(coco_ds, f, protocol=pickle.HIGHEST_PROTOCOL)

    return coco_ds


def test_contrastive_denoising_group_generator():
    cdng_gen = ContrastiveDenoisingGroupGenerator()
