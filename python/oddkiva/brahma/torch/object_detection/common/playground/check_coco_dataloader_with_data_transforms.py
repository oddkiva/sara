# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader

from PySide6.QtCore import Qt

import oddkiva.sara as sara
import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva.sara.dataset.colors import generate_label_colors
from oddkiva.brahma.torch.object_detection.common.data_transforms import (
    ToNormalizedCXCYWHBoxes
)
from oddkiva.brahma.torch.datasets.coco.dataloader import RTDETRImageCollateFunction


def get_coco_dataset() -> coco.COCOObjectDetectionDataset:
    with sara.Timer("COCO Dataset Generation"):
        transform = v2.Compose([
            v2.RandomIoUCrop(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Resize((640, 640)),
            v2.SanitizeBoundingBoxes(),
            # Important: put it after the v2.SanitizeBoundingBoxes.
            ToNormalizedCXCYWHBoxes()
        ])
        coco_ds = coco.COCOObjectDetectionDataset(
            train_or_val='val',
            transform=transform
        )

    return coco_ds


def user_main():
    coco_ds = get_coco_dataset()

    coco_dl = DataLoader(coco_ds,
                         batch_size=16,
                         shuffle=False,
                         collate_fn=RTDETRImageCollateFunction())


    # Font config
    font = sara.make_font()

    # Color config.
    label_colors = generate_label_colors(len(coco_ds.ds.categories))

    sara.create_window(640, 640)
    for batch in coco_dl:
        img, boxes, labels = batch

        N = img.shape[0]

        for n in range(N):
            img_n = img[n]
            boxes_n = boxes[n]
            labels_n = labels[n]

            sara.clear()
            sara.draw_image(img_n.permute(1, 2, 0).contiguous().numpy())

            if len(boxes_n) == 0:
                continue

            # Rescale the bounding boxes.
            whwh = torch.tensor(boxes_n.canvas_size[::-1]).tile(2)[None]
            boxes_rescaled = boxes_n * whwh
            cx, cy, w, h = boxes_rescaled.unbind(-1)
            x = cx - 0.5 * w
            y = cy - 0.5 * h
            xywh = torch.stack((x, y, w, h), dim=-1)

            for box, label in zip(xywh.tolist(), labels_n.tolist()):
                x, y, w, h = box
                label_color = label_colors[label]
                label_name = coco_ds.ds.categories[label].name

                sara.draw_rect((x, y), (w, h), label_color, 2)
                sara.draw_boxed_text(x, y, label_name, label_color, font, angle=0.)

            if sara.get_key() == Qt.Key.Key_Escape:
                return


sara.run_graphics(user_main)
