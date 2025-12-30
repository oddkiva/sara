# copyright (c) 2025 david ok <david.ok8@gmail.com>

import torch

from oddkiva.brahma.torch.losses.focal_loss import focal_loss


class FocalLoss(torch.nn.Module):
    """The focal loss.
    """

    def __init__(self, gamma: float = 2, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def extract_matched_queries(
        self,
        query: torch.Tensor,
        matching: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
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
        # image_ixs = torch.cat([
        #     # the image index n is repeated as many times as there are labeled
        #     # boxes in the training samples
        #     torch.full_like(qixs_matched, n)
        #     for n, (qixs_matched, _) in enumerate(matching)
        # ])
        # query_ixs = torch.cat([qixs_n for (qixs_n, _) in matching])
        # return query[image_ixs, query_ixs]

        return torch.cat([
            query[n][qixs_matched]
            for n, (qixs_matched, _) in enumerate(matching)
        ])

    def extract_matched_targets(
        self,
        targets: list[torch.Tensor],
        matching: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
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
        return torch.cat([
            targets[n][tixs_n]
            for n, (_, tixs_n) in enumerate(matching)
        ])

    def forward(
        self,
        query_scores: torch.Tensor,
        target_scores: list[torch.Tensor],
        matching: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        r"""
        Parameters:
            query_scores:
                score values must be normalized in the range $[0, 1]$.
            target_scores:
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

        # Extract the batches
        qscores_matched = self.extract_matched_queries(query_scores, matching)
        tscores_matched = self.extract_matched_targets(target_scores, matching)

        return focal_loss(qscores_matched, tscores_matched, self.alpha, self.gamma)
