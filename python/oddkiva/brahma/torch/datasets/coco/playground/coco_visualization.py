# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from PySide6.QtGui import Qt
from loguru import logger

import matplotlib.pyplot as plt
import numpy as np

import oddkiva.sara as sara
import oddkiva.brahma.torch.datasets.coco as coco


def generate_label_color_dict(
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
    logger.info('Instantiating COCO DB...')
    db = coco.COCO.instance_train2017()
    ann_db = coco.ImageAnnotationDB(
        coco.COCO.group_annotations_by_image(db),
        db.categories,
        'train'
    )

    label_categories = {
        c.id: c
        for c in db.categories
    }

    # Display config.
    font = sara.make_font()
    label_colors = generate_label_color_dict(db.categories)

    logger.info('Launching visualization window...')
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
           sara.draw_boxed_text(x, y, label_name, label_color, font, angle=0.)

       if sara.get_key() == Qt.Key.Key_Escape:
           break


sara.run_graphics(user_main)
