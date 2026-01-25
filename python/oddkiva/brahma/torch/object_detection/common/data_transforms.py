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


class FromRgb8ToRgb32f(v2.Transform):
    """
    NOTE:
    This is a very hacky and non-robust data transformation that can change any
    tensor data like `BoundingBoxes` objects if we are not careful.
    """

    _transformed_types = (
        torch.Tensor,
    )
    def __init__(self) -> None:
        super().__init__()

    def _transform(self, inpt: Any, _: dict[str, Any]) -> Any:
        if inpt.dtype == torch.uint8:
            inpt = inpt.float() / 255

        return inpt

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._transform(inpt, params)
