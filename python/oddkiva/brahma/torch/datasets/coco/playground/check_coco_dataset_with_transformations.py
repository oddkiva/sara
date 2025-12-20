# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import pickle
from loguru import logger

import torchvision.transforms.v2 as v2

from PySide6.QtCore import Qt

import oddkiva.sara as sara
import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva import DATA_DIR_PATH
from oddkiva.sara.dataset.colors import generate_label_colors


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
                v2.SanitizeBoundingBoxes()
            ])
            coco_ds = coco.COCOObjectDetectionDataset(
                train_or_val='train',
                transform=transform
            )
        with sara.Timer("COCO Dataset Serialization"):
            logger.info( f'Serializing COCO Dataset object to file: {coco_fp}...')
            with open(coco_fp, 'wb') as f:
                pickle.dump(coco_ds, f, protocol=pickle.HIGHEST_PROTOCOL)

    return coco_ds


def user_main():
    FORCE_RECREATE = False
    with sara.Timer("Get or create COCO dataset..."):
        coco_ds = get_or_create_coco_dataset(force_recreate=FORCE_RECREATE)

    font = sara.make_font()
    label_colors = generate_label_colors(len(coco_ds.ds.categories))

    sara.create_window(640, 640)
    for img, boxes, labels in coco_ds:
        sara.clear()
        sara.draw_image(img.permute(1, 2, 0).contiguous().numpy())

        for box, label in zip(boxes.tolist(), labels.tolist()):
            x, y, w, h = box

            label_color = label_colors[label]
            label_name = coco_ds.ds.categories[label].name

            sara.draw_rect((x, y), (w, h), label_color, 2)
            sara.draw_boxed_text(x, y, label_name, label_color, font, angle=0.)

        if sara.get_key() == Qt.Key.Key_Escape:
            break



sara.run_graphics(user_main)
