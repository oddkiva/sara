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

        Parameters
        ----------

        dropout:
            $0.1$ is the default as in the paper.
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
        """
        Parameters
        ----------

        memory:
            enhanced feature maps.
        """
        pass


class MultiScaleDeformableDecoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
