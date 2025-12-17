# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import pickle
from collections.abc import Iterable
from loguru import logger

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontMetrics

import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from torch.types import Number

import oddkiva.sara as sara
import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva import DATA_DIR_PATH
from oddkiva.sara.dataset.colors import generate_label_colors
from oddkiva.brama.torch.datasets.coco.dataloader import collate_fn
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    dn_detr.query_denoiser import ContrastiveDenoisingGroupGenerator


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
                # v2.ToDtype(torch.float32, scale=True),
                # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                v2.Resize((640, 640)),
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

    coco_dl = DataLoader(
        dataset=coco_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )
    img, boxes, labels = iter(coco_dl)
    assert img.shape == (16, 3, 640, 640)
    assert len(boxes) == 16
    assert len(labels) == 16

    cdngg = ContrastiveDenoisingGroupGenerator(80)
    cdng = cdngg.forward(300, {
        'boxes': boxes,
        'labels': labels
    })


    # Display config
    font = make_font()
    label_colors = generate_label_colors(coco_ds.ds.categories)

    sara.create_window(640, 640)
    sara.draw_image(img.permute(1, 2, 0).contiguous().numpy())

    for box, label in zip(boxes.tolist(), labels.tolist()):
        x, y, w, h = box

        label_color = label_colors[label]
        label_name = coco_ds.ds.categories[label].name

        sara.draw_rect((x, y), (w, h), label_color, 2)
        draw_boxed_text(x, y, label_name, label_color, font, angle=0.)

    while sara.get_key() != Qt.Key.Key_Escape:
        logger.debug("Random key pressed...")
        continue
    logger.info("Terminating...")


sara.run_graphics(user_main)
