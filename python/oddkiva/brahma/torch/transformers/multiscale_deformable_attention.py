# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torch
import torch.nn.functional as F


class MultiscaleDeformableAttention(torch.nn.Module):

    def __init__(self, attention_head_count: int,
                 feature_dim: int,
                 value_dim: int,
                 key_count_per_scale: int = 4):
        """Constructs a multiscale deformable attention layer.

        As described in the paper:
        Deformable DETR: Deformable Transformers for End-to-End Object Detection
        <https://arxiv.org/pdf/2010.04159>

        Everything is summarized in the Figure 2 of the paper.

        Args:
            attention_head_count: the dimension of the output value vector.
            value_dim: the dimension of the output value vector.
            key_count_per_scale: the number of keys we want to consider for
                each feature map of the feature pyramid.

        """
        super(MultiscaleDeformableAttention, self).__init__()
        self.attention_head_count = attention_head_count
        self.value_dim = value_dim
        self.key_count_per_scale = key_count_per_scale

        self.value_projector = torch.nn.Linear(feature_dim, value_dim)

        self.positional_embedding_fn = torch.nn.Sequential(
            torch.nn.Linear(2, value_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(value_dim, value_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(value_dim, value_dim),
            torch.nn.ReLU(),
        )

        # As per figure 2 of the paper
        self.sampling_offset_predictors = [
            torch.nn.Linear(feature_dim, key_count_per_scale * 2)
            for _ in range(attention_head_count)
        ]

        self.attention_weight_predictors = [
            torch.nn.Sequential(
                torch.nn.Linear(feature_dim, key_count_per_scale),
                torch.nn.Softmax()
                torch.nn.
            )
            for _ in range(attention_head_count)
        ]

        self.final_projections = [torch.nn.Linear(value_dim, value_dim)
                                 for _ in range(attention_head_count)]

    def normalize_query_positions(
        self,
        X: torch.Tensor,
        query_positions: torch.Tensor
    ) -> torch.Tensor:
        _, h, w, _ = X.shape
        wh_inverse = torch.tensor([1. / w, 1. / h], dtype=torch.float32)
        query_positions_normalized = query_positions * wh_inverse
        return query_positions_normalized

    def form_queries(
        self,
        X: torch.Tensor,
        query_positions: torch.Tensor
    ) -> torch.Tensor:
        X_sampled = X[query_positions]
        positional_embeds = self.positional_embedding_fn(query_positions)
        queries = X_sampled + positional_embeds
        return queries

    def predict_offsets( self, queries: torch.Tensor) -> torch.Tensor:
        """Predicts the position offset for each key and value vectors.

        The offset prediction should take into account:
        - the positional embedding tensor ϕ
        - the feature map X
        """
        # As per figure 2 of the paper
        position_deltas = torch.stack([
            offset_predictor(queries)
            for offset_predictor in self.sampling_offset_predictors
        ], dim=0)

        return position_deltas

    def predict_attention_weights(
        self,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        # Predict the attention weights
        attention_weights = torch.stack([
            attn_weight_pred(queries)
            for attn_weight_pred in self.attention_weight_predictors
        ], dim=0)
        return attention_weights

    def sample_values(self,
                      X: torch.Tensor,
                      query_positions: torch.Tensor,
                      position_deltas: torch.Tensor) -> torch.Tensor:
        # The pairs of (key-value) vectors are normalized in the range [0, 1]
        #
        # The following coordinates are normalized in [0, 1]
        # key_value_positions = query_positions + position_deltas
        #
        # Good: PyTorch has a function `torch.nn.functional.grid_sample` in its
        # API to sample the feature using bilinear interpolation, but they must
        # be in the range [-1, 1], where:
        # - (x=-1, y=-1) is the left-top corner of the feature map
        # - (x=+1, y=+1) is the right-bottom corner of the feature map
        key_value_positions = torch.clamp(
            2. * (query_positions + position_deltas) - 1,
            min = -1., max=1.
        ).unsqueeze(1)
        values = F.grid_sample(X, key_value_positions)
        return values

    def predict_value(self,
                      X: torch.Tensor,
                      query_positions: torch.Tensor) -> torch.Tensor:
        query_positions_normalized = self.normalize_query_positions(
            X, query_positions
        )
        queries = self.form_queries(X, query_positions_normalized)
        position_deltas = self.predict_offsets(queries)

        attn_weigths_sampled = self.predict_attention_weights(queries)
        values_sampled = self.sample_values(X,
                                            query_positions_normalized,
                                            position_deltas)

        # Aggregate by linearly combining the sampled values with the attention
        # weights.
        values_aggregated = torch.sum(attn_weigths_sampled * values_sampled)

        final_value = torch.sum([
            self.final_projections(values_aggregated)
            for _ in range(attention_head_count)
        ])
        return final_value


    def forward(self,
                X: torch.Tensor,
                query_positions: torch.Tensor,
                values: torch.Tensor) -> torch.Tensor:
        pass
