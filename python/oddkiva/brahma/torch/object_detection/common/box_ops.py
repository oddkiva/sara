# copyright (c) 2025 david ok <david.ok8@gmail.com>

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
    lt = boxes[..., :2]
    rb = boxes[..., 2:]
    lt_fixed = torch.min(lt, rb)
    rb_fixed = torch.max(lt, rb)
    boxes_fixed = torch.cat((lt_fixed, rb_fixed), dim=-1)
    return boxes_fixed


def inter_and_union_areas(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    only_compute_diagonal: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters:
        boxes1:
            Tensor of M box coordinates in the format 'left-top-right-bottom'
            with shape (M, 4).
        boxes2:
            Tensor of N box coordinates in the format 'left-top-right-bottom'
            with shape (N, 4).

    Returns:
        inter_area, union_area:
            intersection and union areas for all possible pairs `(b1, b2)`.
            Each of these tensors have shape (M, N).
    """
    assert len(boxes1.shape) == 2
    assert len(boxes2.shape) == 2

    lt1 = boxes1[:, :2]  # Shape is (M, 2)
    lt2 = boxes2[:, :2]  # Shape is (N, 2)
    rb1 = boxes1[:, 2:]  # Shape is (M, 2)
    rb2 = boxes2[:, 2:]  # Shape is (N, 2)

    # Calculate the intersection area for all possible box pairs (b1, b2).
    #
    # The trick is to use broadcasting and by introducing a new axis.
    #
    # The max operation coupled with the broadcasting rule produce a tensor of
    # shape:
    # [max, (M, 1, 2), (N, 2)] --> (M, N, 2)
    if only_compute_diagonal:
        assert boxes1.shape == boxes2.shape
        inter_lt = torch.max(lt1, lt2)
        inter_rb = torch.min(rb1, rb2)
    else:
        inter_lt = torch.max(lt1[:, None, :], lt2)
        inter_rb = torch.min(rb1[:, None, :], rb2)

    inter_wh = (inter_rb - inter_lt).clamp(min=0.)    # (M, N, 2)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]  # (M, N)

    wh1 = boxes1[..., 2:] - boxes1[..., :2]  # (M, 2)
    area1 = wh1[..., 0] * wh1[..., 1]      # (M,)

    wh2 = boxes2[..., 2:] - boxes2[..., :2]  # (M, 2)
    area2 = wh2[..., 0] * wh2[..., 1]      # (M,)

    # With the broadcasting rule
    #            (M, 1)           (1, N)           (M, N)
    if only_compute_diagonal:
        union_area = area1 + area2 - inter_area
    else:
        union_area = area1[:, None] + area2[None, :] - inter_area
        # Shape is (M, N)

    return inter_area, union_area


def smallest_enclosing_box_area(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    only_compute_diagonal: bool = False
) -> torch.Tensor:
    lt1 = boxes1[:, :2]  # Shape is (M, 2)
    lt2 = boxes2[:, :2]  # Shape is (N, 2)
    rb1 = boxes1[:, 2:]  # Shape is (M, 2)
    rb2 = boxes2[:, 2:]  # Shape is (N, 2)

    # Calculate the intersection area for all possible box pairs (b1, b2).
    #
    # The trick is to use broadcasting and by introducing a new axis.
    #
    # The max operation coupled with the broadcasting rule produce a tensor of
    # shape:
    # [max, (M, 1, 2), (N, 2)] --> (M, N, 2)
    if only_compute_diagonal:
        assert boxes1.shape == boxes2.shape
        lt_encl = torch.min(lt1, lt2)
        rb_encl = torch.max(rb1, rb2)
    else:
        lt_encl = torch.min(lt1[:, None, :], lt2)
        rb_encl = torch.max(rb1[:, None, :], rb2)

    wh_encl = (rb_encl - lt_encl).clamp(min=0.) # (M, N, 2)
    a_encl = wh_encl[..., 0] * wh_encl[..., 1]  # (M, N)

    return a_encl
