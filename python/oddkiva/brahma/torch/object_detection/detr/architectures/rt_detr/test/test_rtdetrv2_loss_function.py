# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from loguru import logger

from torch.utils.data import DataLoader
from torchvision.transforms import v2

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch import DEFAULT_DEVICE
import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva.brahma.torch.datasets.coco.dataloader import (
    RTDETRImageCollateFunction
)
from oddkiva.brahma.torch.object_detection.common.data_transforms import (
    ToNormalizedCXCYWHBoxes,
    ToNormalizedFloat32
)
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.config import RTDETRConfig
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.model import RTDETRv2



FORCE_RECREATE = False
COCO_BATCH_SAMPLE = DATA_DIR_PATH / 'coco_batch_sample.pkl'


def get_coco_val_dl():
    logger.info(f"Instantiating COCO dataset...")
    transform = v2.Compose([
        v2.RandomIoUCrop(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Resize((640, 640)),
        v2.SanitizeBoundingBoxes(),
        # Sanitize before the box normalization please.
        ToNormalizedCXCYWHBoxes(),
        ToNormalizedFloat32(),
    ])
    coco_ds = coco.COCOObjectDetectionDataset(
        train_or_val='val',
        transform=transform
    )

    logger.info(f"Instantiating COCO dataloader...")
    coco_dl = DataLoader(
        dataset=coco_ds,
        batch_size=4,
        shuffle=False,
        collate_fn=RTDETRImageCollateFunction()
    )

    return coco_dl

def get_rtdetrv2_model():
    config = RTDETRConfig()
    model = RTDETRv2(config)
    return model


def test_rtdetrv2_loss_function():
    gpu0 = DEFAULT_DEVICE

    coco_val_dl = get_coco_val_dl()

    logger.info(f"Getting first batch sample from COCO dataloader...")
    coco_it = iter(coco_val_dl)

    logger.info(f"Instantiating RT-DETR v2 model...")
    rtdetrv2 = get_rtdetrv2_model()

    rtdetrv2 = rtdetrv2.to(gpu0)

    img, boxes, labels = next(coco_it)
    img = img.to(gpu0)
    boxes = [b.to(gpu0) for b in boxes]
    labels = [l.to(gpu0) for l in labels]
    assert (0 <= img).all() and (img <= 1).all()

    x = img
    targets = {
        'boxes': boxes,
        'labels': labels
    }
    print(boxes)
    y = rtdetrv2.forward(x, targets)
