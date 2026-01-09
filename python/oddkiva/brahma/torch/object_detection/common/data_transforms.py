# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from typing import Any

import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms.v2 as v2
from torchvision.tv_tensors import BoundingBoxFormat, BoundingBoxes


class ToNormalizedCXCYWHBoxes(v2.Transform):

    # The magic recipe
    _transformed_types = (BoundingBoxes,)

    def __init__(self) -> None:
        super().__init__()

    def _transform(self, boxes: BoundingBoxes, _: dict[str, Any]) -> Any:
        in_fmt = boxes.format.value.lower()

        box_data = torchvision.ops.box_convert(
            boxes,
            in_fmt=in_fmt,
            out_fmt='cxcywh'
        )

        hw = boxes.canvas_size
        wh = hw[::-1]
        whwh = torch.tensor(wh).tile(2)[None]  # Shape is (1, 4)
        box_data_normalized = box_data / whwh

        boxes_normalized = BoundingBoxes(
            box_data_normalized,
            format=BoundingBoxFormat.CXCYWH,
            canvas_size=hw
        )

        return boxes_normalized

    def transform(self, boxes: Any, params: dict[str, Any]) -> Any:
        return self._transform(boxes, params)


class ToNormalizedFloat32(v2.Transform):
    _transformed_types = (
        torch.Tensor,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, _: dict[str, Any]) -> Any:  
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        return inpt

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._transform(inpt, params)
