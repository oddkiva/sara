# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from typing import Any

import torch
import torchvision
import torchvision.transforms.v2 as v2
from torchvision.tv_tensors import BoundingBoxFormat, BoundingBoxes


class ToNormalizedCXCYWHBoxes(v2.Transform):

    # The magic recipe
    _transformed_types = (BoundingBoxes,)

    def __init__(self) -> None:
        super().__init__()

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        boxes = inpt
        assert type(boxes) is BoundingBoxes
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
