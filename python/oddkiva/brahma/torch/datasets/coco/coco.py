# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Union

import torch
from torchvision.io import decode_image

from oddkiva import DATA_DIR_PATH


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
        train_or_val: Literal['train', 'val'] = 'train'
    ):
        self.data = data
        if train_or_val == 'train':
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

        def decode_json_array(json_array: list, Class: Any) -> list:
            return [Class(**j) for j in json_array]

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
    def instance_test2017() -> InstanceDB:
        with open(COCO.INSTANCE_TEST2017_FP, 'r') as fp:
            jdict = json.load(fp)
            return COCO.parse_instancedb_json(jdict)

    @staticmethod
    def group_annotations_by_image(db: InstanceDB) -> list:
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

    ObjectDetectionDatasetType = 

    @staticmethod
    def make_object_detection_dataset(
        train_or_val = Union[Literal['train'], Literal['val']]
    ) -> ImageAnnotationDB:
        if dataset_type == 'train':
            db = COCO.instance_train2017()
        elif dataset_type == 'val':
            db = COCO.instance_val2017()
        else:
            raise ValueError(f'COCO Dataset type must be {'train'} or {'val'}')

        annotations = ImageAnnotationDB(
            COCO.populate_image_annotations(db),
            train_or_val
        )
