import json
from pathlib import Path
from typing import Any, Literal
from dataclasses import dataclass

from PySide6.QtGui import QFont, QFontMetrics
import matplotlib.pyplot as plt

import numpy as np

import torch
from torchvision.io import decode_image

from oddkiva import DATA_DIR_PATH
import oddkiva.sara as sara


@dataclass
class Info:
    description: str
    url: str
    version: str
    year: int
    contributor: str
    date_created: str


@dataclass
class License:
    id: int
    name: str
    url: str


@dataclass
class Image:
    id: int
    license: int
    file_name: str
    coco_url: str
    height: int
    width: int
    date_captured: str
    flickr_url: str


@dataclass
class Annotation:
    id: int
    segmentation: Any
    area: float
    iscrowd: int
    image_id: int
    bbox: tuple[float, float, float, float]
    category_id: int


@dataclass
class Category:
    id: int
    supercategory: str
    name: str


@dataclass
class InstanceDB:
    info: Info
    licenses: list[License]
    images: list[Image]
    annotations: list[Annotation]
    categories: list[Category]


@dataclass
class ImageAnnotations:
    image: Image
    annotations: list[Annotation]


class ImageAnnotationDB:

    def __init__(
        self,
        data: list[ImageAnnotations],
        trainval_type: Literal['train', 'val'] = 'train'
    ):
        self.data = data
        if trainval_type == 'train':
            self.image_dir_path = COCO.TRAIN_IMAGES_DIR_PATH
        else:
            self.image_dir_path = COCO.VAL_IMAGES_DIR_PATH

    def __getitem__(self, i: int) -> ImageAnnotations:
        return self.data[i]

    def __len__(self):
        return len(self.data)

    def read_image(self, image: Image) -> torch.Tensor:
        image_filepath = self.image_dir_path / image.file_name
        image_data = decode_image(str(image_filepath))
        return image_data


def decode_json_array(json_array: list, Class: Any) -> list:
    return [Class(**j) for j in json_array]


class COCO:

    MAIN_DIR_PATH: Path = DATA_DIR_PATH / 'coco'
    ANNOTATIONS_DIR_PATH = MAIN_DIR_PATH / 'annotations'

    INSTANCE_TRAIN2017_FP = ANNOTATIONS_DIR_PATH / 'instances_train2017.json'
    INSTANCE_VAL2017_FP = ANNOTATIONS_DIR_PATH / 'instances_val2017.json'

    TRAIN_IMAGES_DIR_PATH = MAIN_DIR_PATH / 'train2017'
    VAL_IMAGES_DIR_PATH = MAIN_DIR_PATH / 'val2017'

    assert MAIN_DIR_PATH.exists()
    assert ANNOTATIONS_DIR_PATH.exists()
    assert TRAIN_IMAGES_DIR_PATH.exists()
    assert VAL_IMAGES_DIR_PATH.exists()

    @staticmethod
    def parse_instancedb_json(jdict: dict[str, Any]) -> InstanceDB:
        info = Info(**jdict['info'])
        licenses = decode_json_array(jdict['licenses'], License)
        images = decode_json_array(jdict['images'], Image)
        annotations = decode_json_array(jdict['annotations'], Annotation)
        categories = decode_json_array(jdict['categories'], Category)

        data = InstanceDB(
            info,
            licenses,
            images,
            annotations,
            categories
        )
        return data

    @staticmethod
    def instance_train2017() -> InstanceDB:
        with open(COCO.INSTANCE_TRAIN2017_FP, 'r') as fp:
            jdict = json.load(fp)
            return COCO.parse_instancedb_json(jdict)

    @staticmethod
    def instance_val2017() -> InstanceDB:
        with open(COCO.INSTANCE_VAL2017_FP, 'r') as fp:
            jdict = json.load(fp)
            return COCO.parse_instancedb_json(jdict)

    @staticmethod
    def populate_image_annotations(db: InstanceDB) -> list:
        image_dict = {
            im.id: im
            for im in db.images
        }

        annotations_grouped_by_image_id: dict[int, list[Annotation]] = {}
        for a in db.annotations:
            if a.image_id not in annotations_grouped_by_image_id:
                annotations_grouped_by_image_id[a.image_id] = []
            anns = annotations_grouped_by_image_id[a.image_id]
            anns.append(a)

        image_annotations = [
            ImageAnnotations(
                image_dict[image_id],
                annotations_grouped_by_image_id[image_id]
            )
            for image_id in annotations_grouped_by_image_id
        ]

        return image_annotations


def user_main():
    db = COCO.instance_train2017()
    ann_db = ImageAnnotationDB(
        COCO.populate_image_annotations(db),
        'train'
    )

    # Color config.
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(db.categories)))
    colors = (colors[:, :3] * 255).astype(np.int32)

    # Font config
    font = QFont()
    font_size = 12
    italic, bold, underline = False, True, False
    font.setPointSize(font_size)
    font.setItalic(italic)
    font.setBold(bold)
    font.setUnderline(underline)

    font_metrics = QFontMetrics(font)
    text_x_offset = 1
    def calculate_text_box_size(text: str):
        label_text_rect = font_metrics.boundingRect(text)
        size = label_text_rect.size()
        w, h = size.width() + text_x_offset * 2 + 1, size.height()
        return w, h

    label_categories = {
        c.id: c
        for c in db.categories
    }
    label_colors = {
        db.categories[i].id: colors[i]
        for i in range(len(db.categories))
    }

    sara.create_window(640, 640)
    for a in ann_db:
       im = ann_db.read_image(a.image)

       sara.clear()
       sara.draw_image(im.permute(1, 2, 0).numpy())
       for ann in a.annotations:
           x, y, w, h = ann.bbox

           cat_id = ann.category_id
           color = label_colors[cat_id]
           sara.draw_rect((x, y), (w, h), color, 2)

           label_name = label_categories[cat_id].name

           w, h = calculate_text_box_size(label_name)
           l, t = (int(x), int(y))
           sara.fill_rect((l - text_x_offset, int(t - h)), (w, h), color)
           sara.draw_text((l, t - 2), label_name, (0, 0, 0), font_size, 0.,
                          italic, bold, underline)

       sara.get_key()


sara.run_graphics(user_main)
