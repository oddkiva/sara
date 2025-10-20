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


class CAVIAR(ClassificationDatasetABC):

    def __init__(
        self,
        root_path: Path,
        transform: Optional[v2.Transform] = None
    ):
        self._root_path = root_path
        self._transform = transform

        # Populate the list of image paths
        self._image_paths = [
            root_path / f for f in os.listdir(root_path)
            if '.jpg' in str(f)
        ]
        self._image_paths.sort()

        def extract_image_label(p: Path) -> str:
            return p.name[:4]

        self._image_labels = [extract_image_label(p) for p in self._image_paths]
        self._image_class_ids = [int(l) - 1 for l in self._image_labels]

        self._labels = list(set(self._image_class_ids))
        self._labels.sort()

        self._classes = [int(image_label) - 1
                         for image_label in self._image_labels]

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
