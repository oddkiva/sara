# copyright (c) 2025 david ok <david.ok8@gmail.com>

import torch
import torch.nn.functional as F

from oddkiva.brahma.torch.losses.focal_loss import focal_loss


class FocalLoss(torch.nn.Module):
    """The focal loss class.

    This focal loss also penalizes non-matched query boxes. Because they are
    not matched by the Hungarian matcher, non-matched query boxes are not
    assigned with object class ID.
    """

    def __init__(self, gamma: float = 2, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def extract_matched_query_indices(
        self,
        matching: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Params:
            matching:
                This is the optimal matching computed by the Hungarian
                assignment method.

                The matching is list of pairs of tensors (`query_ixs`,
                `target_ixs`):

                - `len(matches)` is the number of samples in the batch;

                - the pair `(query_ixs, target_ixs)` are 1D-tensors of
                  identical size, which is the number of labeled boxes in the
                  training sample.
        """
        image_ixs = torch.cat([
            torch.full_like(qixs_n, n)
            for n, (qixs_n, _) in enumerate(matching)
        ]).to(torch.int64)

        box_ixs = torch.cat([
            qixs_n
            for (qixs_n, _) in matching
        ]).to(torch.int64)

        return image_ixs, box_ixs

    def forward(
        self,
        query_scores: torch.Tensor,
        target_labels: list[torch.Tensor],
        matching: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        r"""
        Parameters:
            query_scores:
                score values must be normalized in the range $[0, 1]$.
            target_labels:
                zeros or ones.
            matching:
                This is the optimal matching computed by the Hungarian
                assignment method.

                The matching is list of pairs of tensors (`query_ixs`,
                `target_ixs`):

                - `len(matches)` is the number of samples in the batch;

                - the pair `(query_ixs, target_ixs)` are 1D-tensors of
                  identical size, which is the number of labeled boxes in the
                  training sample.
        """

        N, top_K, num_classes = query_scores.shape

        # Extract the indices of the matched object queries.
        #
        # An object query is indexed by the pair `(n, k)` where `n` is the
        # image index in the batch and `k` the query index for this image.
        qixs_matched = self.extract_matched_query_indices(matching)

        # NOTE: we want to penalize the non-matched boxes.
        #
        # 1. Form the tensor of ground-truth labels (N, K)
        #    A non-matched query boxes is assigned with the `non-object` class
        #    whose index is `num_classes`.
        non_object_idx = num_classes
        tlabels = torch.full((N, top_K), non_object_idx, dtype=torch.int64)
        # 2. Collect the list of matched ground-truth box indices.
        #    That list of indices is a permutation of ${0, 1, 2, ..., N_n}$.
        tlabels_matched = torch.cat([
            tlabels_n[tixs_n]
            for tlabels_n, (_, tixs_n) in zip(target_labels, matching)
        ]).to(torch.int64)
        # 3. Replace the `non-object` class with the appropriate ground-truth
        #    class IDs.
        tlabels[qixs_matched] = tlabels_matched
        # Shape is (N, top_K)
        #
        # 4. Use the built-in function to constitute the (N, K, C) tensor of
        #    target probabilities.
        tscores = F.one_hot(tlabels, num_classes=num_classes+1)[..., :-1]
        # Shape is (N, top_K, C), not (N, top_K, C+1)!

        return focal_loss(query_scores, tscores, self.alpha, self.gamma)
