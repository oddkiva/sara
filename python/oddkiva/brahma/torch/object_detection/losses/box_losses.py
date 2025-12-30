# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torch as T

from oddkiva.brahma.torch.object_detection.common.box_ops import (
    inter_and_union_areas,
    smallest_enclosing_box_area
)


def iou(boxes1: T.Tensor, boxes2: T.Tensor,
        eps: float = 1e-8) -> T.Tensor:
    r"""
    Calculates the ratio of the intersection over the union, aka the Jaccard
    index for all possible pairs of boxes $(i, j)$ where:

    - $i \in \{0, M-1 \}$ and $M = \mathrm{Card}(B_1)$ is the number of boxes
      in `boxes1`
    - $j \in \{0, N-1 \}$ and $N = \mathrm{Card}(B_2)$ is the number of boxes
      in `boxes2`

    Returns:
        `J`:
            A matrix $\mathbf{J} = (J_{ij})$ of shape $(M, N)$, where
            $$
              J_{i,j} =
              \frac{\mathrm{area}(\mathbf{b}_i \cap \mathbf{b}_j)}
                   {\mathrm{area}(\mathbf{b}_i \cup \mathbf{b}_j)}.
            $$
    """
    inter, union = inter_and_union_areas(boxes1, boxes2)
    return inter / union.clamp(min=eps)


def penalized_iou(boxes1: T.Tensor, boxes2: T.Tensor,
                  eps: float = 1e-8) -> T.Tensor:
    """
    This is the actual generalized IoU (https://giou.stanford.edu/).
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    a_inter, a_union = inter_and_union_areas(boxes1, boxes2)
    a_encl = smallest_enclosing_box_area(boxes1, boxes2)

    # The classical IoU score.
    iou = a_inter / a_union.clamp(min=eps)

    # The gap area in the enclosed bounding box but not in the union is the
    # penalty. It should be as possible.
    #
    # When it becomes it means that the intersection is the union.
    gap_area = 1 - a_union / a_encl.clamp(min=eps)

    giou = iou - gap_area
    return giou


def giou(boxes1: T.Tensor, boxes2: T.Tensor, normalize: bool,
         eps: float = 1e-8) -> T.Tensor:
    r"""
    Actually... the generalized IoU (https://giou.stanford.edu/) could be
    re-explained with an alternative interpretation.

    The generalized IoU (gIoU) is basically the sum of two scores:

    - the classical IoU score if boxes overlap ranging in $[0, 1]$.

    - the IoU score between (1) the union of two boxes $b_1$ and $b_2$ and the
      smallest enclosing bounding box enclosing $b_1$ and $b_2$ also ranging in
      $[0, 1]$.

      The paper says that we want to *minimize* the the normalized "gap area"
      that is not covered in the union of the box $C$:
      $$
        \frac{\mathrm{area}(C \setminus A \cup B)}{\mathrm{area}(C)}
      $$

      But this exactly equivalent to *maximizing*:
      $$
        \frac{\mathrm{area}(A \cup B)}{\mathrm{area}(C)}
      $$

      And this is exactly how my intuition was and not as the penalized IoU
      score that the authors saw it when they presented the gIoU.

    Instead of penalizing the classical IoU score with that normalized gap area,
    we want to maximize two types of IoU scores as the same time.

    This is better in terms of interpretation, because the gIoU score stays
    positive and in the range $[0, 2]$. Then by simply dividing it by 2, the gIoU
    score is normalized in the range $[0, 1]$.

    If the IoU is $0$, then this second score does indeed comes to the rescue and
    penalizes the gap area in the enclosed bounding box.

    That is why I don't see much benefit in offsetting with that constant
    $-1$, the original expose complicates more than the natural intuition that
    I get.

    We cannot unsee how awkward the original implementation from a numerical
    point of view:
    ```
    iou - (area - union) / area

    # This turns out to be awkward and simplifies as:
    # iou - 1 + union/area
    #
    # So I will just do my own way...
    ```

    Likewise, the gIoU loss score would simply be: `1 - 0.5 * gIoU`.

    The original implementation designs it as: $2 - \mathrm{gIoU}$ which
    actually ranges in $[0, 2]$. This is something to keep in mind if we want
    to be absolutely careful when designing composite loss functions.

    To finish, notice that the gIoU is pretty much very equivalent to the cIoU
    formulation because it also means that we want to minimize the distance
    between the two box centers and maximizes the similarity between the
    similarity between their aspect ratio.

    The superiority of the gIoU is evident as computationally cheaper. In
    contrast, the cIoU involves the `arctan` operation to measure the similarity
    between the aspect ratio.
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    a_inter, a_union = inter_and_union_areas(boxes1, boxes2)

    a_encl = smallest_enclosing_box_area(boxes1, boxes2)

    # The classical IoU score.
    iou_1 = a_inter / a_union.clamp(min=eps)
    # The second IoU score between the box union and the enclosing box.
    iou_2 = a_union / a_encl.clamp(min=eps)

    giou = iou_1 + iou_2
    if normalize:
        return giou * 0.5
    else:
        return giou


def loss_iou(boxes1: T.Tensor, boxes2: T.Tensor,
             eps: float = 1e-8) -> T.Tensor:
    return 1 - loss_iou(boxes1, boxes2, eps=eps)


def loss_giou(boxes1: T.Tensor, boxes2: T.Tensor,
              eps: float = 1e-8) -> T.Tensor:
    return 1 - giou(boxes1, boxes2, normalize=True, eps=eps)
