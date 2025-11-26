# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiscaleDeformableAttention(nn.Module):
    """ This class implements the multiscale deformable attention layer as
    described in the paper [Deformable DETR: Deformable Transformers for
    End-to-End Object Detection](https://arxiv.org/pdf/2010.04159).

    Everything is summarized in the Figure 2 of the paper.
    """

    def __init__(self,
                 embed_dim: int,
                 attention_head_count: int,
                 value_dim: int,
                 pyramid_level_count: int = 4,
                 kv_count_per_level: int = 4,
                 sampling_offset_scale_factor: float = 0.5):
        """Constructs a multiscale deformable attention layer.

        Parameters:
            embed_dim:
                the dimension of the query and key vectors.
            value_dim:
                the dimension of the value vectors.
            attention_head_count:
                the dimension of the output value vector.
            pyramid_level_count:
                the number of levels to use from the feature pyramid, from the last
                one.

                Suppose that a CNN backbone produces a feature pyramid with 5
                levels where at each level $l$, we denote the final feature map by
                $\\mathbf{F}_l$.

                If, for example, pyramid_level_count is set to 3, then we use only
                the last 3 feature maps of the feature pyramid
                $\\mathbf{F}_3, \\mathbf{F}_4, \\mathbf{F}_5$.

            kv_count_per_level:
                the number of key-value pairs we want to consider for each feature
                map of the feature pyramid.

            sampling_offset_scale_factor:
                yet another hyperparameter.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.attention_head_count = attention_head_count
        self.pyramid_level_count = pyramid_level_count
        self.kv_count_per_level = kv_count_per_level
        self.kv_count_per_attention_head = \
            pyramid_level_count * kv_count_per_level
        self.kv_count_per_query = \
            attention_head_count * self.kv_count_per_attention_head

        # As described in figure 2 of the paper, the sampling offset is simply
        # a linear predictor.
        #
        # Notice that we could have done the following:
        # self.sampling_offset_funcs = [
        #     nn.Linear(embed_dim, key_count_per_scale * 2)
        #     for _ in range(attention_head_count)
        # ]
        #
        # But the reference implementation uses equivalently a single linear
        # predictor and this is more efficient as we do a single call.
        self.sampling_offset_predictors = nn.Linear(
            embed_dim,
            self.kv_count_per_query * 2,
        )
        self.sampling_offset_scale_factor = sampling_offset_scale_factor

        # Likewise, we learn a single linear predictor
        self.attn_weight_predictors = nn.Linear(
            embed_dim,
            self.kv_count_per_query
        )
        # We will need to reshape the sampling offset predictions to reorder
        # things after that.
        # N, L, d_K = x.shape
        # N, L, H * K = y.shape
        #
        # Carefully apply the reshape function and the softmax over the last
        # dimension.
        # torch.reshape(y, (N, L, H, K))
        # torch.softmax(y, dim=-1)

        self.value_projector = nn.Linear(embed_dim, value_dim)
        self.final_projections = nn.ModuleList(
            nn.Linear(value_dim, value_dim)
            for _ in range(attention_head_count)
        )

    def predict_positional_offsets(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Predicts the relative position deltas for each key-value pairs w.r.t
        the query position.

        We implement it as per figure 2 of the paper.
        """
        batch_size, query_count, _ = queries.shape

        position_deltas = self.sampling_offset_predictors(queries)
        position_deltas = torch.reshape(
            position_deltas,
            (batch_size, query_count,
             self.attention_head_count,
             self.kv_count_per_attention_head, 2)
        )

    def predict_attention_weights(
        self,
        queries: torch.Tensor
    ) -> torch.Tensor:
        """
        Predicts the attention weights for each key-value pairs w.r.t. each
        query.

        We implement it as per figure 2 of the paper.
        """
        attn_weights = self.attn_weight_predictors(queries)

        batch_size, query_count, _ = queries.shape
        attn_weights = torch.reshape(
            attn_weights,
            (batch_size, query_count,
             self.attention_head_count,
             self.kv_count_per_attention_head)
        )
        attn_weights = torch.softmax(attn_weights, dim=-1)

        return attn_weights

    def sample_values(
        self,
        query_encodings: torch.Tensor,
        key_value_positions: torch.Tensor,
    ) -> torch.Tensor:
        # The positions of the (key-value) pairs are normalized in the range
        # [0, 1].
        #
        # Good: PyTorch has a function `torch.nn.functional.grid_sample` in its
        # API to sample the feature using bilinear interpolation, but they must
        # be in the range [-1, 1], where:
        # - (x=-1, y=-1) is the left-top corner of the feature map
        # - (x=+1, y=+1) is the right-bottom corner of the feature map
        x_kv = torch.clamp(
            2. * key_value_positions - 1,
            min = -1., max=1.
        ).unsqueeze(1)
        values = torch.cat([
            F.grid_sample(q, x_kv)
            for q in query_encodings
        ])
        return values

    def forward(self,
                queries: torch.Tensor,
                query_geometry_logits: torch.Tensor,
                value: torch.Tensor,
                value_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Predict the attention weights between the K best keys in the L image
        # levels for each query q.
        w_lkq = self.predict_attention_weights(queries)

        # Predict the positional deltas of the K best keys in the L image
        # levels for each query q.
        Δx_lkq = self.predict_positional_offsets(queries)

        box_q = F.sigmoid(query_geometry_logits)
        wh_q = box_q[:, :, :2]
        cxcy_q = box_q[:, :, :2]
        x_q = cxcy_q

        # The scale factor in the original code has gotten me raising a few
        # eyebrows... Why can't the sampling offset predictor learn it?
        #
        # Is there a clear interpretable reason as for why the Deformable-DETR
        # authors think it should be rescaled by the query box width and
        # height? My own interpretation is as follows.
        #
        # TODO: can we improve the predictor to avoid these multiplications?
        crazy_scale_factor = wh_q * num_points_scale * \
            self.sampling_offset_scale_factor
        Δx_lkq = Δx_lkq * crazy_scale_factor
        # The final position of the key-value pairs
        x_lkq = x_q + Δx_q

        if value_mask is None:
            value_masked = value
        else:
            value_masked = value_mask.to(dtype=value.dtype) * value
        value_lkq = self.sample_values(value_masked, x_lkq)

        # Aggregate by linearly combining the sampled values with the attention
        # weights.
        value_q = torch.sum(w_lkq * value_lkq)

        value_q = torch.sum([
            self.final_projections(values_aggregated)
            for _ in range(self.attention_head_count)
        ])

        return value_q
