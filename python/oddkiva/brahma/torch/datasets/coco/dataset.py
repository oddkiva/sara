# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from typing import Any, Literal, Optional

import torch
from torch.utils.data import Dataset
import torchvision.tv_tensors as tvt
import torchvision.transforms.v2 as v2

from oddkiva.brahma.torch.datasets.coco import COCO, Annotation


DatasetType = Literal['train', 'val']
ImageAnnotations = dict[str, list[Any]]


class COCOObjectDetectionDataset(Dataset):

    def __init__(
        self,
        train_or_val: DatasetType = 'train',
        transform: Optional[v2.Transform] = None
    ):
        self.ds = COCO.make_object_detection_dataset(train_or_val)
        self.transform = transform

        self.category_id_to_label_idx_map = {
            c.id: i
            for i, c in enumerate(self.ds.categories)
        }

    def __len__(self):
        return len(self.ds)

    def _vectorize_annotations(
        self,
        annotations: list[Annotation]
    ) -> tuple[list[tuple[float, float, float, float]], list[int]]:
        return (
            [ann.bbox for ann in annotations],
            [ann.category_id for ann in annotations]
        )


    def read_annotated_image(
        self, idx: int
    ) -> tuple[torch.Tensor, tvt.BoundingBoxes, torch.Tensor]:
        labeled_image = self.ds[idx]

        image = self.ds.read_image(labeled_image.image)
        boxes, categories = self._vectorize_annotations(labeled_image.annotations)
        _, h, w = image.shape

        boxes = tvt.BoundingBoxes(
            torch.tensor(boxes),
            format=tvt.BoundingBoxFormat.XYWH,
            canvas_size=(h, w)
        )

        labels = torch.tensor([
            self.category_id_to_label_idx_map[category_id]
            for category_id in categories
        ], dtype=torch.int32)

        return image, boxes, labels


    def __getitem__(
        self,
        idx: int
    ) -> tuple[torch.Tensor, tvt.BoundingBoxes, torch.Tensor]:
        image, boxes, labels = self.read_annotated_image(idx)

        # In order to use v2.Transforms.
        ann_dict = {
            'boxes': boxes,
            'labels': labels
        }
        if self.transform is not None:
            image, ann_dict = self.transform(
                image,
                ann_dict
                # self  # For the mosaic data transform.
            )
        # Unwrap the dictionary please...
        boxes = ann_dict['boxes']
        labels = ann_dict['labels']

        return image, boxes, labels
