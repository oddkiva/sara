from collections import OrderedDict

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from oddkiva.brahma.torch.backbone.multi_layer_perceptron import (
    MultiLayerPerceptron
)
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.multiscale_deformable_attention import (
        MultiscaleDeformableAttention
    )


class BoxGeometryEmbeddingMap(MultiLayerPerceptron):

    def __init__(self, embed_dim: int):
        super().__init__(4, 2 * embed_dim, embed_dim, 2, activation='relu')
        self._reinitialize_learning_parameters()

    def _reinitialize_learning_parameters(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)


class BoxObjectClassLogitHead(nn.Linear):

    def __init__(self,
                 embed_dim: int,
                 num_classes: int,
                 initial_prob: float = 0.1):
        super().__init__(embed_dim, num_classes)
        nn.init.constant_(
            self.bias,
            self._initial_class_logit_value(initial_prob)
        )

    def _initial_class_logit_value(self, prob: float) -> float:
        """
        The interpretation is the bias is a low starting probability value that
        any anchor with box size of 5% of the image size contains an object of
        any class.
        """
        logit_value = -math.log((1 - prob) / prob)
        return logit_value

class BoxGeometryLogitHead(MultiLayerPerceptron):

    def __init__(self,
                 embed_dim: int,
                 layer_count: int,
                 activation: str = 'relu'):
        super().__init__(embed_dim, embed_dim, 4, layer_count,
                         activation=activation)
        self._reinitialize_learning_parameters()

    def _reinitialize_learning_parameters(self):
        # Only reinitialize the last layer.
        nn.init.constant_(self.layers[-1].weight, 0)
        nn.init.constant_(self.layers[-1].bias, 0)


class MultiScaleDeformableTransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        image_level_count: int,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        normalize_before: bool = False,
        training: bool = False
    ):
        """Constructs the base layer of a Transformer Decoder block with
        reasonable default parameters.

        Parameters:
            embed_dim:
                The output embedding feature dimension.
            num_heads:
                The number of attention heads in each decoder layer.
            feedforward_dim:
                The output feature dimension of the feed-forward network.
            dropout:
                The dropout probability value.
            normalize_before:
                Optionally choose to apply the layer norm operation:

                1. *before* applying the attention layer, and
                2. *before* applying the feed-forward network.

                By default, the layer norm operation is applied:

                1. *after* the attention layer, and
                2. *after* the feed-forward network.
        """
        super().__init__()

        self.normalize_before = normalize_before

        self.self_attention = torch.nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout,
            batch_first=True
        )

        self.dropout_1 = torch.nn.Dropout(p=dropout)
        self.layer_norm_1 = torch.nn.LayerNorm(embed_dim)

        self.cross_attention = MultiscaleDeformableAttention(
            embed_dim, num_heads, embed_dim, image_level_count
        )
        self.dropout_2 = torch.nn.Dropout(p=dropout)
        self.layer_norm_2 = torch.nn.LayerNorm(embed_dim)

        self.feedforward = torch.nn.Sequential(OrderedDict([
            ("linear-1", torch.nn.Linear(embed_dim, feedforward_dim)),
            ("activation", torch.nn.ReLU()),
            ("dropout", torch.nn.Dropout(p=dropout)),
            ("linear-2", torch.nn.Linear(feedforward_dim, embed_dim))
        ]))
        self.dropout_3 = torch.nn.Dropout(p=dropout)
        self.layer_norm_3 = torch.nn.LayerNorm(embed_dim)

        self.training = training

    def with_positional_embeds(self,
                       queries: torch.Tensor,
                       positional_embeds: torch.Tensor | None) -> torch.Tensor:
        if positional_embeds is None:
            return queries
        else:
            return queries + positional_embeds

    def forward(
        self,
        query_embeds: torch.Tensor,
        query_geometry: torch.Tensor,
        memory: torch.Tensor,
        query_positional_embeds: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""
        Decodes the object query matrix $\mathbf{Q}$ into an output value matrix
        $\mathbf{V}^{+}$.

        - The memory is a peculiar term that simply denotes the 2D feature maps
          $\mathbf{F}$, which, when flattened, corresponds to the value matrix
          $\mathbf{V}$.

        - Denoting the position of the query vector $\mathbf{q}$ by $\mathbf{x}$,
          we calculate its positional embedding $\phi(\mathbf{x})$ and we
          obtain the key matrix $\mathbf{K} = \mathbf{F} + \phi(\mathbf{X})$.

        In the end, we calculate the decoded object queries as

        $$
        \mathbf{V}^+ = \sigma \left( \frac{1}{\sqrt{d_k}}
            \mathbf{Q} (\mathbf{F} + \phi(\mathbf{X})^\intercal
        \right) \mathbf{V}
        $$

        Parameters:
            query_embeds:
                The query encoding stacked as row vectors
            query_geometry:
                The 4D box geometry for each object query row vectors of
                $\mathbf{Q}$.
            memory:
                the concatenated query vectors that are calculated from the
                feature pyramid
                (backbone -> AIFI -> CCFF -> projection -> concatenation).

        Returns:
            Decoded queries $\mathbf{V}^+$
        """

        # IMPORTANT
        # ---------
        #
        # In RT-DETR, a query is considered to be a new independent input even
        # if it is actually produced by the CNN backbone extractor and the
        # encoder.
        assert query_embeds.requires_grad is False
        assert query_geometry.requires_grad is False

        # Prepare the data.
        Q = K = self.with_positional_embeds(query_embeds, query_positional_embeds)
        V = query_embeds

        # 1. Self-Attention -> (Dropout -> Add -> Norm).
        #
        # 1.1. Self-attention.
        ΔV, _ = self.self_attention.forward(Q, K, value=V, attn_mask=attn_mask)
        # 1.2.a) Perturb the enhanced value residuals to avoid overfitting in the
        #        self-attention block with the dropout layer.
        # 1.2.b) Apply the (Add -> Norm) layer.
        V_super = self.layer_norm_1(V + self.dropout_1(ΔV))

        # 2. Cross-attention -> (Dropout -> Add -> Norm)
        ΔV = self.cross_attention.forward(
            self.with_positional_embeds(V_super, query_positional_embeds),
            query_geometry_logits,
            memory,
            memory_mask
        )
        V_super = self.layer_norm_2(V_super + self.dropout_2(ΔV))

        # 3. FFN -> (Dropout -> Add -> Norm)
        ΔV = self.feedforward(V_super)
        V_super = self.layer_norm_3(V_super + self.dropout_3(ΔV))

        return V_super


class MultiScaleDeformableTransformerDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        kv_count_per_level: list[int],
        # QUERY SELECTION
        num_classes: int = 80,
        # ATTENTION PARAMS
        attn_head_count: int = 8,
        attn_feedforward_dim: int = 2048,
        attn_num_layers: int = 6,
        attn_dropout: float = 0.1,
        normalize_before: bool = False,
    ):
        super().__init__()

        # ---------------------------------------------------------------------
        # OBJECT QUERY DECODING
        # ---------------------------------------------------------------------
        self.decoder = nn.ModuleList([
            MultiScaleDeformableTransformerDecoderLayer(
                hidden_dim, attn_head_count, len(kv_count_per_level),
                attn_feedforward_dim, attn_dropout,
                normalize_before=normalize_before
            )
            for _ in range(attn_num_layers)
        ])

        # These two heads are mainly there to accelerate the training
        # convergence
        #
        # Query class logits
        self.decoder_class_logits_head = nn.ModuleList(
            nn.Linear(hidden_dim, num_classes) for _ in range(attn_num_layers)
        )
        # Query geometry logits
        self.decoder_box_geometry_head = nn.ModuleList(
            MultiLayerPerceptron(hidden_dim, hidden_dim, 4, 3)
            for _ in range(attn_num_layers)
        )

        self.box_geometry_embedding_map: BoxGeometryEmbeddingMap
        self.box_geometry_logit_heads: list[BoxGeometryLogitHead]
        self.box_class_logit_heads: list[BoxObjectClassLogitHead]

    def forward(
        self,
        query: torch.Tensor, query_geometry_logits: torch.Tensor,
        value: torch.Tensor,
        value_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert query.requires_grad is False
        assert query_geometry_logits.requires_grad is False

        # Get the actual query geometry by activating the logits with the
        # sigmoid function.

        # Initialize:
        # - the current query embedding
        # - the current query geometry logits (the object geometry before the
        #   sigmoid activation)
        # - the current query geometry (the object box geometry)
        query_curr = query
        query_geom_logits_curr = query_geometry_logits
        query_geom_curr = F.sigmoid(query_geom_logits_curr)

        query_next: torch.Tensor | None = None
        query_class_logits_next: torch.Tensor | None = None
        query_geom_logits_next: torch.Tensor | None = None
        query_geom_next: torch.Tensor | None = None

        query_geometries_denoised = []
        query_class_logits_denoised = []

        for i, decoder_layer in enumerate(self.decoder):
            assert type(decoder_layer) is MultiScaleDeformableTransformerDecoderLayer

            # Calculate the corresponding embed vector of the box geometry
            query_geom_embed_curr = self.box_geometry_embedding_map\
                .forward(query_geom_curr)

            # Denoise the current query.
            query_next = decoder_layer.forward(
                query_curr, query_geom_curr, value,
                query_positional_embeds=query_geom_embed_curr,
                attn_mask=value_mask, memory_mask=None
            )

            # Estimate the new object class logits (object class probabilities).
            query_class_logits_next = self.box_class_logit_heads[i](query_next)

            # Estimate the new object geometry (cx cy w h).
            Δ_query_geom_logits = self.box_geometry_logit_heads[i](query_next)
            query_geom_logits_next = \
                query_geom_logits_curr + Δ_query_geom_logits
            query_geom_next = F.sigmoid(query_geom_logits_next)

            # Store the denoised results.
            query_geometries_denoised.append(query_geom_next)
            query_class_logits_denoised.append(query_class_logits_next)

            # Update for the next denoising iteration.
            query_curr = query_next
            query_geom_logits_curr = query_geom_logits_next
            query_geom_curr = query_geom_next

        return (torch.stack(query_geometries_denoised),
                torch.stack(query_class_logits_denoised))
