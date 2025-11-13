from collections import OrderedDict

import torch

from oddkiva.brahma.torch.object_detection\
    .detr.rtdetr.multiscale_deformable_attention import (
        MultiscaleDeformableAttention
    )


class MultiScaleDeformableTransformerDecoderLayer(torch.nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
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
            embed_dim, num_heads
        )
        self.dropout_2 = torch.nn.Dropout(p=dropout)
        self.layer_norm_2 = torch.nn.LayerNorm(embed_dim)

        self.feedforward = torch.nn.Sequential(OrderedDict([
            ("dec-linear-1", torch.nn.Linear(embed_dim, feedforward_dim)),
            ("dec-activation", torch.nn.ReLU()),
            ("dec-dropout", torch.nn.Dropout(p=dropout)),
            ("dec-linear-2", torch.nn.Linear(feedforward_dim, embed_dim))
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
        pass


class MultiScaleDeformableDecoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
