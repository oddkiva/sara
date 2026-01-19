# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import numpy as np

from pydantic import BaseModel, TypeAdapter, field_serializer

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image


class BBox(NamedTuple):
    xmin: int
    ymin: int
    xmax: int
    ymax: int

class Category(BaseModel):
    token: str
    name: str
    description: str

class Mask(BaseModel):
    size: Tuple[int, int]
    counts: str

class ObjectAnnotation(BaseModel):
    token: str
    category_token: str
    bbox: BBox
    mask: Mask | None
    attribute_tokens: List[str]
    sample_data_token: str

class Sample(BaseModel):
    token: str
    timestamp: int
    log_token: str
    key_camera_token: str

    @field_serializer('timestamp')
    def serialize_dt(self, dt: datetime):
        return int(dt.timestamp() * 1e6)

    @property
    def datetime(self):
        return datetime.fromtimestamp(self.timestamp / 1e6, tz=timezone.utc)

class SampleData(BaseModel):
    token: str
    ego_pose_token: str
    calibrated_sensor_token: str
    filename: str
    fileformat: str
    width: int
    height: int
    timestamp: int
    is_key_frame: bool
    prev: str
    next: str

    @property
    def datetime(self):
        return datetime.fromtimestamp(self.timestamp / 1e6, tz=timezone.utc)


class NuImagesDataset(Dataset):

    def __init__(self, metadata_dirpath: Path, sample_dirpath: Path):
        self._metadata_dirpath = metadata_dirpath
        self._sample_dirpath = sample_dirpath
        self._category_fp = self._metadata_dirpath / 'category.json'
        self._object_ann_fp = self._metadata_dirpath / 'object_ann.json'
        self._sample_fp = self._metadata_dirpath / 'sample.json'
        self._sample_data_fp = self._metadata_dirpath / 'sample_data.json'

        self._category = self._read_json(self._category_fp,
                                         TypeAdapter(List[Category]))
        self._object_ann = self._read_json(self._object_ann_fp,
                                           TypeAdapter(List[ObjectAnnotation]))
        self._sample = self._read_json(self._sample_fp,
                                       TypeAdapter(List[Sample]))
        self._sample_data = self._read_json(self._sample_data_fp,
                                            TypeAdapter(List[SampleData]))

        self._category_dict = {cat.token: cat for cat in self._category}
        self._sample_data_dict = {s.token: s for s in self._sample_data}

        # The list of annotated images
        self._annotated_images: Dict[str, List[ObjectAnnotation]] = {}
        for obj_ann in self._object_ann:
            token = obj_ann.sample_data_token
            if token in self._annotated_images:
                self._annotated_images[token].append(obj_ann)
            else:
                self._annotated_images[token] = [obj_ann]

    def _read_json(self, fp: Path, type_adapter):
        json_str = fp.read_text()
        return type_adapter.validate_json(json_str)

    def __len__(self):
        return len(self._sample)

    def category(self, obj_ann: ObjectAnnotation) -> Category:
        return self._category_dict[obj_ann.category_token]

    def sample_data(self, token: str) -> SampleData:
        return self._sample_data_dict[token]

    def sample_image_path(self, token: str) -> Path:
        return self._sample_dirpath / self._sample_data_dict[token].filename

    def sample_image_data(self, token: str) -> torch.Tensor:
        return decode_image(self.sample_image_path(token),
                            mode=ImageReadMode.RGB)


def show_annotation(dataset: NuImagesDataset,
                    sample_data_token: str,
                    annotation: List[ObjectAnnotation]) -> np.array:
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()

    sample_image_data = dataset.sample_image_data(sample_data_token)
    ax.imshow(sample_image_data.permute(1, 2, 0))

    for ann in annotation:
        xmin, ymin, xmax, ymax = ann.bbox
        xs = [xmin, xmax, xmax, xmin, xmin]
        ys = [ymin, ymin, ymax, ymax, ymin]
        ax.plot(xs, ys, color='red')

        category = dataset.category(ann)
        ax.text(xmin, ymin, category.name, color='red')

    plt.show()


if __name__ == '__main__':
    drive_path = Path('/media/Linux Data')
    nuimages_dirpath = drive_path / 'nuimages'
    nuimages_all_metadata = nuimages_dirpath  / 'nuimages-v1.0-all-metadata'
    nuimages_sample_dirpath = drive_path / 'nuimages'
    nuimages_v1_mini = nuimages_all_metadata / 'v1.0-mini'

    dataset = NuImagesDataset(nuimages_v1_mini, nuimages_sample_dirpath)

    sample_data_token = list(dataset._annotated_images)[0]

    sample_image_path = dataset.sample_image_path(sample_data_token)
    assert sample_image_path.exists()

    sample_image_data = dataset.sample_image_data(sample_data_token)

    annotation = dataset._annotated_images[sample_data_token]

    show_annotation(dataset, sample_data_token, annotation)

    import IPython; IPython.embed()
