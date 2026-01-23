# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torchvision.transforms.v2 as v2

from PySide6.QtCore import Qt

import oddkiva.sara as sara
import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva.brahma.torch.object_detection.common.mosaic import Mosaic
from oddkiva.sara.dataset.colors import generate_label_colors


def get_coco_dataset() -> coco.COCOObjectDetectionDataset:
    transform = v2.Compose([
        Mosaic(use_cache=False),
        v2.RandomIoUCrop(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.SanitizeBoundingBoxes()
    ])
    coco_ds = coco.COCOObjectDetectionDataset(
        train_or_val='val',
        transform=transform
    )

    return coco_ds


def user_main():
    with sara.Timer("Get COCO dataset..."):
        coco_ds = get_coco_dataset()

    font = sara.make_font()
    label_colors = generate_label_colors(len(coco_ds.ds.categories))

    sara.create_window(1024, 1024)
    for img, boxes, labels in coco_ds:
        print(img)
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
