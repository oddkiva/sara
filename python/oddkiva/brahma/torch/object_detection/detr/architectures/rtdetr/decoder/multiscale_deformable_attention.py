# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from typing import Iterable

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiscaleDeformableAttention(nn.Module):
    """ This class implements the multiscale deformable attention layer as
    described in the paper [Deformable DETR: Deformable Transformers for
    End-to-End Object Detection](https://arxiv.org/pdf/2010.04159).

    Everything is summarized in the Figure 2 of the paper -- except the
    peculiar rescaling of the positional offsets of sample positions.
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
                the dimension of the value vectors for *each attention head*.
                Notice the important wording *for each attention head*.
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
        self.kv_count_per_level_inverse = 1. / kv_count_per_level
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
        self.sampling_offset_max_value = sampling_offset_scale_factor

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

        self.value_projector = nn.Linear(
            embed_dim,
            value_dim * attention_head_count
        )
        self.final_projections = nn.Linear(
            value_dim * attention_head_count,
            embed_dim
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
        # Reshape appropriately to normalize with respect to the LK attention
        # weights.
        attn_weights = attn_weights.reshape(
            batch_size, query_count,
            self.attention_head_count,
            self.kv_count_per_attention_head  # L * K = 3 * 4
        )
        # Softmax on the last dimension.
        attn_weights = torch.softmax(attn_weights, dim=-1)
        assert len(attn_weights.shape) == 4

        return attn_weights

    def predict_positional_offsets(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Predicts the relative position deltas for each key-value pairs w.r.t
        the query position.

        We implement it as per figure 2 of the paper.
        """
        batch_size, query_count, _ = queries.shape

        position_deltas = self.sampling_offset_predictors(queries)
        position_deltas = position_deltas.reshape(
            batch_size, query_count,
            self.attention_head_count,
            self.kv_count_per_attention_head,
            2
        )
        assert len(position_deltas.shape) == 5

        return position_deltas

    def calculate_sample_positions(
        self,
        query_geometry_logits: torch.Tensor,
        Δx_lkq: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Returns the locations $(x_lkq, y_lkq)$ of the key-value pairs for each
        query $q$.

        The location is normalized in the range [0, 1], w.r.t. the image sizes
        $(w_l, h_l)$ of the feature pyramid at level $l$ for learnability
        reasons.

        This is a bit tricky implementation because it performs some surprising
        rescaling operations, which were not explained in the paper.
        """

        # Activate the query geometries logits to obtain the query geometries.
        # The query geometry is nothing less than the geometry of the object
        # bounding box $(x_q, y_q, w_q, h_q)$, where:
        #
        # - $(x_q, y_q)$ is the box center, and
        # - $(w_q, h_q)$ is the box sizes.
        #
        # Let us just denote by $b_q$ the object boxes
        assert len(query_geometry_logits.shape) == 3
        assert query_geometry_logits.shape[2] == 4
        assert query_geometry_logits.shape[:2] == Δx_lkq.shape[:2]
        N, top_K, _ = query_geometry_logits.shape

        b_q = F.sigmoid(query_geometry_logits)

        # The list of box sizes $(w_q, h_q)$
        wh_q = b_q[:, :, 2:]
        # The list of box centers $(x_q, y_q)$
        xy_q = b_q[:, :, :2]
        # Let's shorten the box centers as
        x_q = xy_q

        # The scale factor in the original code has gotten me raising a few
        # eyebrows... Why can't the sampling offset predictor learn it?
        #
        # Is there a clear interpretable reason as for why the Deformable-DETR
        # authors think it should be rescaled by the query box width and
        # height? My own interpretation is as follows.
        #
        # TODO: can we improve the predictor to avoid these multiplications?
        the_crazy_scale_factor = wh_q * \
            self.kv_count_per_level_inverse * \
            self.sampling_offset_max_value
        assert the_crazy_scale_factor.shape == (N, top_K, 2)

        Δx_lkq = Δx_lkq * the_crazy_scale_factor[:, :, None, None, :]
        # The final position of the key-value pairs
        x_lkq = x_q[:, :, None, None, :] + Δx_lkq

        # Check the dimensionality of position
        N, top_K, _ = b_q.shape
        assert x_lkq.shape == (N, top_K,
                               self.attention_head_count,
                               self.kv_count_per_attention_head,
                               2)

        return x_lkq

    def sample_values(
        self,
        values: torch.Tensor,
        value_pyramid_hw_sizes: list[torch.Size],
        value_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Implementation note:
        This method contains a set of technical tensor manipulation.

        The "trick" is to permute the tensor axes and reshape the dimensions in
        such a way that we collapse the batch index and the attention head
        index into a single 1D index.

        This is so that we can use the built-in function
        `torch.nn.functional.grid_sample` in the 2D case.
        """

        # Shorten the variable names.
        N, _, d_k = values.shape
        C = values.shape[2]
        M = self.attention_head_count
        L = self.pyramid_level_count
        K = self.kv_count_per_level
        d_v = self.value_dim
        top_K = value_positions.shape[1]

        # Check that we are feeding the inputs with appropriate dimensions.
        assert len(value_pyramid_hw_sizes) == L
        assert C == M * d_v

        # Reconstruct the pyramid of query maps from its flattened tensor of
        # queries.
        #
        # 1. Calculate the different strides for image level.
        pyr_strides = [0] + [
            h * w
            for (h, w) in value_pyramid_hw_sizes
        ]
        pyr_strides = [*itertools.accumulate(pyr_strides)]

        # 2.   Reconstruct the pyramid of query maps.
        # 2.1. Separate the tensor into a list of flattened query maps, each
        #      one of these corresponding to the image level l and reconstruct
        #      the pyramid of query maps.
        # 2.2. Reshape the flattened query maps into 2D maps of feature
        #      vectors.
        # 2.3. Permute the axes so that we can sample values with the built-in
        #      function `torch.nn.functional.grid_sample(...)`.

        ixs_start = pyr_strides[:-1]
        ixs_end = pyr_strides[1:]

        # A value pyramid is a pyramid of 2D value maps, each one of them of
        # shapes (N, C, H_l, W_l)
        value_pyramid = [
            values[:, s:e, :]\
            # Reconstruct the value tensor as a 2D maps
            .reshape(N, H_l, W_l, d_k)\
            # Rearrange the data as (N, C, H_l, W_l)
            .permute(0, 3, 1, 2)
            # Reorganize the axes as follows:
            #     (    N,       C, H_l, W_l)
            #  == (    N, M * d_v, H_l, W_l)   (cf. asserts)
            #  -> (N * M,     d_v, H_l, W_l)
            #
            # The idea is to treat the values as a mini-batch of
            # lower-dimensional values, each one of them produced by the
            # attention heads.
            #
            # Notice the underlying 1D array has not changed in any case.
            .reshape(N * M, d_v, H_l, W_l)
            for (s, e, (H_l, W_l)) in zip(ixs_start, ixs_end,
                                          value_pyramid_hw_sizes)
        ]

        # 3. Split the list of key-value positions per image level.
        #
        #    `x_kv_per_level` is the list of value sample locations for each
        #    image level `l`.
        #
        #    x_per_level[l] has shape (N, top-K, M * K, 2).
        x_per_level = [
            value_positions[:, :, :, K*i:K*(i+1), :]\
            .permute(0, 2, 1, 3, 4)\
            .reshape(N * M, top_K, K, 2)
            # The steps are so that :
            # 0. shape: (N, top-K, M, K, 2)
            #            0      1  2  3  4
            # 1. shape: (N, M, top-K, K, 2)
            # 2. shape: (N * M, top-K, K, 2)

            for i in range(len(value_pyramid_hw_sizes))
        ]

        values_per_level = []
        for x_l, query_map in zip(x_per_level, value_pyramid):
            # The positions of the (key-value) pairs are normalized in the range
            # [0, 1].
            #
            # Good: PyTorch has a function `torch.nn.functional.grid_sample` in its
            # API to sample the feature using bilinear interpolation, but they must
            # be in the range [-1, 1], where:
            # - (x=-1, y=-1) is the left-top corner of the feature map
            # - (x=+1, y=+1) is the right-bottom corner of the feature map
            x_rescaled = torch.clamp(
                2. * x_l - 1,
                min = -1., max=1.
            )
            # Collapse the pair of indices (attention head index, key index)
            # into a 1D index.
            values_at_level_l = F.grid_sample(query_map, x_rescaled)
            # Shape is (N * M, d_v, top-K, K)

            # Make sure we permute the axes again to perform the attention
            # calculus.
            # 0. Shape is (N * M, d_v, top-K, K)
            # 1. Shape is (N, M, d_v, top-K, K)
            #              0  1    2      3  4
            # 2. Shape is (N, top-K, M, K, d_v)
            values_at_level_l = values_at_level_l\
                .reshape(N, M, d_v, top_K, K)\
                .permute(0, 3, 1, 4, 2)
            values_per_level.append(values_at_level_l)

        values = torch.cat(values_per_level, dim=3)
        assert values.shape == (N, top_K, M, L * K, d_v)
        return values

    def forward(self,
                queries: torch.Tensor,
                query_geometry_logits: torch.Tensor,
                value: torch.Tensor,
                value_spatial_sizes: list[Iterable[int]],
                value_mask: torch.Tensor | None = None) -> torch.Tensor:
        n, top_k, d_k = queries.shape
        M = self.attention_head_count
        LK = self.kv_count_per_attention_head

        # Predict the attention weights between the K best keys in the L image
        # levels for each query q.
        w_qlk = self.predict_attention_weights(queries)
        assert w_qlk.shape == (n, top_k, M, LK)

        # Predict the positional deltas of the K best keys in the L image
        # levels for each query q.
        Δx_qlk = self.predict_positional_offsets(queries)
        assert Δx_qlk.shape == (n, top_k, M, LK, 2)

        # Obtain the final positions of the key-value pairs.
        x_qlk = self.calculate_sample_positions(
            query_geometry_logits, Δx_qlk
        )
        assert x_qlk.shape == (n, top_k, M, LK, 2)


        # TODO: would it better to project the values **after** sampling? This
        # would be **a lot less computations**.
        value_projected = self.value_projector(value)
        # Zero out the masked attention values.
        if value_mask is None:
            value_masked = value
        else:
            value_masked = value_mask.to(dtype=value.dtype) * value_projected
        value_qlk = self.sample_values(value_masked,
                                       value_spatial_sizes,
                                       x_qlk)

        assert value_qlk.shape == (n, top_k, M, LK, d_k)


        # Aggregate by linearly combining the sampled values with the attention
        # weights.
        # Summation over the 3rd axis which represents the pair of
        # indices (image level l, key-value index k) that is collapsed into a
        # single 1D index.
        value_q = torch.sum(w_qlk[..., None] * value_qlk, dim=3)

        # The value is the refined object query vector.
        value_q = self.final_projections(value_q)

        return value_q
