# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from PySide6.QtGui import QFont, QFontMetrics

from torch.types import Number

import oddkiva.sara as sara
import oddkiva.brahma.torch.datasets.coco as coco


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
) -> dict[int, np.ndarray]:
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, len(categories)))
    colors = (colors[:, :3] * 255).astype(np.int32)

    label_colors = {
        categories[i].id: colors[i]
        for i in range(len(categories))
    }

    return label_colors


def user_main():
    db = coco.COCO.instance_train2017()
    ann_db = coco.ImageAnnotationDB(
        coco.COCO.group_annotations_by_image(db),
        'train'
    )

    # Font config
    font = make_font()

    label_categories = {
        c.id: c
        for c in db.categories
    }
    # Color config.
    label_colors = generate_label_colors(db.categories)

    sara.create_window(640, 640)
    for a in ann_db:
       im = ann_db.read_image(a.image)

       sara.clear()
       sara.draw_image(im.permute(1, 2, 0).numpy())
       for ann in a.annotations:
           x, y, w, h = ann.bbox

           cat_id = ann.category_id
           label_color = label_colors[cat_id]
           label_name = label_categories[cat_id].name

           sara.draw_rect((x, y), (w, h), label_color, 2)
           draw_boxed_text(x, y, label_name, label_color, font, angle=0.)

       sara.get_key()


sara.run_graphics(user_main)
