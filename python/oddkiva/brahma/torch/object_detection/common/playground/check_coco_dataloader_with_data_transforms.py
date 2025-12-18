# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from PySide6.QtGui import QFont, QFontMetrics

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from torch.types import Number

import oddkiva.sara as sara
import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva.brahma.torch.object_detection.common.data_transforms import (
    ToNormalizedCXCYWHBoxes
)
from oddkiva.brahma.torch.datasets.coco.dataloader import RTDETRImageCollateFunction


def make_font(font_size: int = 12,
              italic: bool = False,
              bold: bool = True,
              underline: bool = False) -> QFont:
    font = QFont()
    font.setPointSize(font_size)
    font.setItalic(italic)
    font.setBold(bold)
    font.setUnderline(underline)
    return font


def draw_boxed_text(x: Number, y: Number, text: str,
                    box_color: Iterable[Number],
                    font: QFont,
                    text_color: Iterable[Number] = (0, 0, 0),
                    angle: float = 0.) -> None:
    font_metrics = QFontMetrics(font)
    text_x_offset = 1
    def calculate_text_box_size(text: str):
        label_text_rect = font_metrics.boundingRect(text)
        size = label_text_rect.size()
        w, h = size.width() + text_x_offset * 2 + 1, size.height()
        return w, h

    w, h = calculate_text_box_size(text)
    l, t = (int(x), int(y))
    sara.fill_rect((l - text_x_offset, int(t - h)), (w, h), box_color)

    sara.draw_text((l, t - 2 * text_x_offset), text, text_color,
                   font.pointSize(), angle, font.italic(), font.bold(),
                   font.underline())


def generate_label_colors(
    categories: list[coco.Category],
    colormap: str = 'rainbow'
) -> np.ndarray:
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, len(categories)))
    colors = (colors[:, :3] * 255).astype(np.int32)
    return colors


def get_coco_dataset() -> coco.COCOObjectDetectionDataset:
    with sara.Timer("COCO Dataset Generation"):
        transform = v2.Compose([
            # v2.RandomIoUCrop(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Resize((640, 640)),
            v2.SanitizeBoundingBoxes(),
            # Important: put it after the v2.SanitizeBoundingBoxes.
            ToNormalizedCXCYWHBoxes()
        ])
        coco_ds = coco.COCOObjectDetectionDataset(
            train_or_val='train',
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
    font = make_font()

    # Color config.
    label_colors = generate_label_colors(coco_ds.ds.categories)

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
                draw_boxed_text(x, y, label_name, label_color, font, angle=0.)

            sara.get_key()


sara.run_graphics(user_main)

