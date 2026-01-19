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


class LaST(ClassificationDatasetABC):

    def __init__(
        self,
        root_path: Path,
        transform: Optional[v2.Transform] = None,
        dataset_type: str = 'train'
    ):
        if dataset_type != 'train':
            raise NotImplementedError()

        self._data_dir_path = root_path / dataset_type
        self._transform = transform

        # Populate the list of class directory paths
        self._image_paths = [
            [Path(dir_path) / fn for fn in files if '.jpg' in str(fn)]
            for dir_path, _, files in os.walk(self._data_dir_path)
        ]
        # Flatten the list of list of files
        self._image_paths = sum(self._image_paths, [])
        assert all([fp.exists() for fp in self._image_paths])

        image_labels = [p.parent.name for p in self._image_paths]

        self._class_names = list(set(image_labels))
        self._class_ixs = {
            class_name: i
            for i, class_name in enumerate(self._class_names)
        }
        self._image_label_ixs = [
            self._class_ixs[class_name]
            for class_name in image_labels
        ]

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        image = decode_image(str(self._image_paths[idx]))
        if self._transform is not None:
            image_transformed = self._transform(image)
        else:
            image_transformed = image
        label = self._image_label_ixs[idx]
        return image_transformed, label

    @property
    def classes(self):
        return self._class_names

    @property
    def class_count(self):
        return len(self._class_names)

    @property
    def image_class_ids(self) -> List[int]:
        return self._image_label_ixs

    def image_class_name(self, idx: int) -> str:
        class_id = self._image_label_ixs[idx]
        return self._class_names[class_id]
