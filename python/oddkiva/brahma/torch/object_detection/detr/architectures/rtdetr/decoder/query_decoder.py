from collections import OrderedDict

import torch
import torch.nn as nn

from oddkiva.brahma.torch.backbone.multi_layer_perceptron import (
    MultiLayerPerceptron
)
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.multiscale_deformable_attention import (
        MultiscaleDeformableAttention
    )


class MultiScaleDeformableTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 image_level_count: int,
                 feedforward_dim: int = 2048,
                 dropout: float = 0.1,
                 normalize_before: bool = False):
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

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        memory: list[torch.Tensor]
    ) -> torch.Tensor:
        r"""Decodes the query matrix $\mathbf{Q}$ into an output value matrix
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
            query:
                The query matrix $\mathbf{Q}$
            query_pos:
                The 2D coordinates of each query vectors of the matrix
                $\mathbf{Q}$.
            memory:
                the feature maps.

        Returns:
            Decoded queries $\mathbf{V}^+$
        """

        # IMPORTANT: In RT-DETR, a query is considered to be a new independent
        # input even if it is actually produced by the CNN backbone extractor
        # and the encoder.
        assert query.requires_grad is False


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

    # def _reinitialize_learning_parameters(self):
    #     if self.learn_query_content:
    #         init.xavier_uniform_(self.tgt_embed.weight)

    #     for m in self.input_proj:
    #         init.xavier_uniform_(m[0].weight)

    def forward(self, feature_pyramid: list[torch.Tensor]) -> list[torch.Tensor]:
        raise NotImplementedError()
