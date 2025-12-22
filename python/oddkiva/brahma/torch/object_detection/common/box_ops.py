# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torch


def from_cxcywh_to_ltrb_format(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    l = cx - 0.5 * w
    r = cx + 0.5 * w
    t = cy - 0.5 * h
    b = cy + 0.5 * h
    return torch.stack((l, t, r, b), dim=-1)


def from_ltrb_to_cxcywh_format(boxes: torch.Tensor) -> torch.Tensor:
    l, t, r, b = boxes.unbind(-1)
    cx = 0.5 * (l + r)
    cy = 0.5 * (t + b)
    w = r - l
    h = b - t
    return torch.stack((cx, cy, w, h), dim=-1)


def fix_ltrb_coords(boxes: torch.Tensor) -> torch.Tensor:
    l = boxes[..., 0]
    t = boxes[..., 1]
    r = boxes[..., 2]
    b = boxes[..., 3]
    boxes_fixed = torch.zeros_like(boxes)
    boxes_fixed[..., 0] = torch.min(l, r)
    boxes_fixed[..., 1] = torch.min(t, b)
    boxes_fixed[..., 2] = torch.max(l, r)
    boxes_fixed[..., 3] = torch.max(t, b)
    return boxes_fixed
