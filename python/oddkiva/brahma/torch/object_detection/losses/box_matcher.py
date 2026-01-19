# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torch
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

from oddkiva.brahma.torch.object_detection.common.box_ops import (
    from_cxcywh_to_ltrb_format
)
from oddkiva.brahma.torch.object_detection.losses.box_losses import giou


class BoxMatcher(torch.nn.Module):
    r"""For each training image in the training batch indexed by $n$, we compute
    the optimal corresponce between the query box $i$ and ground-truth/target
    box $j$ based on some cost matrix.

    Let us clarify a bit.

    The cost matrix $C$ is the distance matrix where each coefficient $C_{ij}$
    encodes the distance value between any pair of (query-box, target-box),
    respectively indexed by $i$ and $j$.

    The distance score is the opposite of some similarity score.

    That means the smaller the cost $C_{ij}$, the smaller the distance, the
    more similar the query box `i` is to the target box `j`.

    The cost matrix is a composite cost built from 3 cost matrices:

    - The cost matrix $C_\textrm{class}$ that measures the probability that
      query box `i` has the same object class `target_label[j]` as the target
      box `j`.
    - $C_\textrm{\ell_1}$ that measures the geometric similarity between query
      box `i` and target box `j`, where the box geometry is encoded a 4D
      vector $(c_x, c_y, w, h)$.
    - $C_\textrm{gIoU}$ that measures the generalized IoU score query
      box `i` and target box `j`.

    As we explain in more details in the implementation, we try to avoid writing
    Python for loops as much as possible.

    In doing so, there will be lots of impossible cases, where we estimate a
    distance between the query box from training image `i` to a target box in
    a different training image `j`. Such cases will be filtered out later.
    """

    def __init__(
        self,
        use_focal_loss: bool = True,
        alpha: float = 0.25,
        gamma: float = 2.0,
        weights: dict[str, float] = {
            'class': 2.0,
            'l1': 5.0,
            'giou': 2.0
        }
    ):
        super().__init__()

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        self.w_class = weights['class']
        self.w_box_l1 = weights['l1']
        self.w_box_giou = weights['giou']
        self.eps = 1e-8

    def calculate_class_labeling_cost_matrix(
        self,
        query_label_probs_flat: torch.Tensor,
        tgt_labels_flat: torch.Tensor
    ) -> torch.Tensor:
        if self.use_focal_loss:
            # We operate on the whole batch of training images where we have:
            # - `N * top_K` query boxes in total in the whole batch.
            # - `tgt_count` target boxes in total in the whole batch
            #
            # That means there will be lots of impossible cases, where we
            # estimate a distance between the query box from training image `i`
            # to a target box in training image `j`.
            #
            # These impossible cases will be filtered out later, but we do this
            # because we try to avoid writing loops in Python.
            tgt_class_probs = query_label_probs_flat[:, tgt_labels_flat]

            # The distance between query box `i` and target box `j` can be
            # measured in terms of the focal loss. There are 2 components in the
            # focal loss: the positive part and the negative part.
            #
            # The positive part of the focal loss is a penalty score for
            # labeling the query box `i` with the same object class of target
            # box `j` which is `tgt_labels_flat[j]`.
            #
            # This value approaches 0 as the probability approaches 1.
            fl_pos_component = self.alpha * \
                ((1 - tgt_class_probs) ** self.gamma) * \
                (-torch.log(tgt_class_probs + self.eps))

            # The negative part of the focal loss is the penalty score for NOT
            # labeling query box `i` to the same object class of target box
            # `j`.
            #
            # This value approaches 0 when the probability is close to 1.
            # Conversely this value approaches `-infty` when the probability
            # approaches 0.
            # In other words, the negative part of the focal loss is a
            # **similarity** score that shoots to `-infty`.
            fl_neg_component = (1 - self.alpha) * \
                (tgt_class_probs ** self.gamma) * \
                (-torch.log(1 - tgt_class_probs + self.eps))

            # The composite labeling cost for is the difference.
            #
            # This distance shoots to `-infty` when the estimated probability
            # approaches 1. Likewise, when the estimated probability is poorly
            # estimated, the labeling cost is close to `+infty` because of the
            # positive part of the focal loss.
            C_class = fl_pos_component - fl_neg_component
        else:
            # Straightforward:
            # -> A probability score is an affinity value.
            # -> The opposite of the affinity value is akin to a distance
            #    value.
            #
            # This is less powerful than the focal loss because values ranges
            # in [0, 1].
            C_class = -query_label_probs_flat[:, tgt_labels_flat]

        # Clearly assigning the wrong class label to a predicted box makes us
        # pay an infinitely costlier than getting the geometry wrong, because
        # the two types of geometric costs are both ranging in [0, 1].
        #
        # That means that the training stage will first optimize the the object
        # class label assignment well before optimizing the predicted box
        # geometries.

        return C_class


    @torch.no_grad()
    def forward(
        self,
        query_class_logits: torch.Tensor,
        query_boxes: torch.Tensor,
        target_labels: list[torch.Tensor],
        target_boxes: list[torch.Tensor],
    ):
        N, top_K = query_class_logits.shape[:2]

        # We flatten the pair of batch index and query index.
        if self.use_focal_loss:
            query_class_probs = F.sigmoid(query_class_logits.flatten(0, 1))
        else:
            query_class_probs = query_class_logits.flatten(0, 1).softmax(-1)
        assert query_class_probs.shape == (N * top_K, query_class_logits.shape[-1])

        query_boxes_flat = query_boxes.flatten(0, 1)
        assert query_boxes_flat.shape == (N * top_K, 4)

        # Tensorize the list of target labels and list of target boxes.
        tgt_count_per_sample = [len(boxes) for boxes in target_boxes]
        tgt_count = sum(tgt_count_per_sample)
        tgt_labels_flat = torch.cat([ls for ls in target_labels])
        tgt_boxes_flat = torch.cat([bs for bs in target_boxes])
        assert tgt_count == tgt_labels_flat.shape[0]
        assert tgt_count == tgt_boxes_flat.shape[0]
        assert len(tgt_labels_flat.shape) == 1
        assert tgt_boxes_flat.shape[1] == 4

        query_boxes = from_cxcywh_to_ltrb_format(query_boxes)

        C_class = self.calculate_class_labeling_cost_matrix(
            query_class_probs, tgt_labels_flat
        )
        assert C_class.shape == (N * top_K, tgt_count)

        # Compute the L1 cost between boxes.
        #
        # Note that are lots of unnecessary computations because the authors
        # choose to not make any distinction where the predicted box and the
        # target box are within the sample.
        #
        # The authors find it simpler to make everything as a single GPU calls.
        # But we pay quite a big cost in terms of readability.

        # Everything ranges in [0, 1].
        C_box_l1 = torch.cdist(query_boxes_flat,  # (N * top_K, 4)
                               tgt_boxes_flat,    # (tgt_count, 4)
                               p=1)
        assert C_box_l1.shape == (N * top_K, tgt_count)

        # Make the giou score range in [0, 1].
        C_box_giou = -giou(
            from_cxcywh_to_ltrb_format(query_boxes_flat),
            from_cxcywh_to_ltrb_format(tgt_boxes_flat),
            normalize=True
        )
        assert C_box_giou.shape == (N * top_K, tgt_count)

        # Composite cost matrix
        C = \
            self.w_class * C_class + \
            self.w_box_l1 * C_box_l1 + \
            self.w_box_giou * C_box_giou

        # Finally we extract the relevant cost sub-matrices corresponding to
        # each training image in the batch.
        C = C.reshape(N, top_K, -1).cpu()
        Cs = [
            C[n]
            for n, C in enumerate(C.split(tgt_count_per_sample, -1))
        ]

        # Because we calculated all possible costs w.r.t. every target boxes in
        # every sample
        indices = [linear_sum_assignment(Cn) for Cn in Cs]
        indices = [(torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64))
                   for i, j in indices]

        return indices
