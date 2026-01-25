# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from loguru import logger

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader

import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva.brahma.torch.object_detection.common.mosaic import (
    Mosaic
)

def get_coco_val_dataset():
    logger.info(f"Instantiating COCO dataset...")
    transform = v2.Compose([
        Mosaic(
            output_size=320,
            rotation_range=10,
            translation_range=(0.1, 0.1),
            scaling_range=(0.5, 1.5),
            probability=1.0,
            fill_value=0,
            use_cache=False,
            max_cached_images=50,
            random_pop=True
        ),
        v2.Resize((640, 640)),
        v2.SanitizeBoundingBoxes()
    ])
    ds = coco.COCOObjectDetectionDataset(
        train_or_val='val',
        transform=transform
    )
    logger.info(f"Number of samples: {len(ds)}")

    return ds


def test_mosaic():
    ds = get_coco_val_dataset()

    print(ds[0])

    # logger.info(f"Instantiating COCO dataloader...")
    # dl = DataLoader(
    #     dataset=ds,
    #     batch_size=16,
    #     shuffle=False,
    #     collate_fn=collate_fn
    # )
