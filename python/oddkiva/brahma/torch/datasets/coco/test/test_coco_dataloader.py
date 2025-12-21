# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from loguru import logger

from torch.utils.data import DataLoader
from torchvision.transforms import v2

from oddkiva import DATA_DIR_PATH
import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva.brahma.torch.datasets.coco.dataloader import (
    RTDETRImageCollateFunction
)
from oddkiva.brahma.torch.object_detection.common.data_transforms import (
    ToNormalizedCXCYWHBoxes
)


FORCE_RECREATE = False
COCO_BATCH_SAMPLE = DATA_DIR_PATH / 'coco_batch_sample.pkl'


def test_coco_dataloader():
    logger.info(f"Instantiating COCO dataset...")
    transform = v2.Compose([
        v2.RandomIoUCrop(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Resize((640, 640)),
        ToNormalizedCXCYWHBoxes(),
        v2.SanitizeBoundingBoxes()
    ])
    coco_ds = coco.COCOObjectDetectionDataset(
        train_or_val='val',
        transform=transform
    )

    logger.info(f"Instantiating COCO dataloader...")
    coco_dl = DataLoader(
        dataset=coco_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=RTDETRImageCollateFunction()
    )

    logger.info(f"Getting first batch sample from COCO dataloader...")
    coco_it = iter(coco_dl)
    for _ in range(10):
        img, boxes, labels = next(coco_it)
        assert len(img) == 16
        assert len(boxes) == 16
        assert len(labels) == 16

        hw = img.shape[2:]
        for b in boxes:
            assert b.canvas_size == hw
