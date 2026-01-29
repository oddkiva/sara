# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from loguru import logger

import torch
import torchvision.transforms.v2 as v2

import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva.brahma.torch.object_detection.common.data_transforms import (
    FromRgb8ToRgb32f,
    ToNormalizedCXCYWHBoxes
)
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
        v2.SanitizeBoundingBoxes(),
        ToNormalizedCXCYWHBoxes(),
        FromRgb8ToRgb32f()
    ])
    ds = coco.COCOObjectDetectionDataset(
        train_or_val='val',
        transform=transform
    )
    logger.info(f"Number of samples: {len(ds)}")

    return ds


def test_mosaic():
    ds = get_coco_val_dataset()

    img, boxes, labels = ds[0]

    # Some basic checks.
    assert img.shape == (3, 640, 640)
    assert img.dtype == torch.float32
    assert (0 <= img).all().item() is True and (img <= 1).all().item() is True
    assert len(boxes) == len(labels)

    # Check that the boxes don't have crazily small coordinates.
    assert ((boxes * 640).mean(0) > 20.).all().item() is True
