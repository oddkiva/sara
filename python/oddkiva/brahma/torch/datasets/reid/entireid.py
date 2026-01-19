# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.torch_version
import torchvision.transforms.v2 as v2
if torch.torch_version.TorchVersion(torch.__version__) < (2, 6, 0):
    from torchvision.io.image import read_image as decode_image
else:
    from torchvision.io.image import decode_image

from oddkiva.brahma.common.classification_dataset_abc import (
    ClassificationDatasetABC
)


class ENTIReID(ClassificationDatasetABC):

    def __init__(
        self,
        root_path: Path,
        transform: Optional[v2.Transform] = None,
        dataset_type: str = 'test'
    ):
        if dataset_type == 'train':
            raise NotImplementedError(
                'The dataset owner has not released it yet as of August 2025'
            )

        self._root_path = root_path
        self._transform = transform

        self._data_dir_path = root_path / f"bounding_box_{dataset_type}" 

        # Populate the list of image paths
        self._image_paths = [
            self._data_dir_path/ filename
            for filename in os.listdir(self._data_dir_path)
            if '.jpg' in filename
        ]
        self._image_paths.sort()

        def extract_image_label(p: Path) -> str:
            return p.name[:5]

        self._image_labels = [extract_image_label(p) for p in self._image_paths]
        self._image_class_ids = [int(l) for l in self._image_labels]

        self._labels = list(set(self._image_class_ids))
        self._labels.sort()

        self._classes = [int(l) for l in self._labels]

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        image = decode_image(str(self._image_paths[idx]))
        if self._transform is not None:
            image_transformed = self._transform(image)
        else:
            image_transformed = image
        label = self._image_class_ids[idx]
        return image_transformed, label

    @property
    def classes(self):
        return self._labels

    @property
    def class_count(self):
        return len(self._labels)

    @property
    def image_class_ids(self) -> List[int]:
        return self._image_class_ids

    def image_class_name(self, idx: int) -> str:
        return self._image_labels[idx]
