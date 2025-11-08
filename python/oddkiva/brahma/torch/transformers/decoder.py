# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import copy
from collections import OrderedDict

import torch


class TransformerDecoderLayer(torch.nn.Module):
    """This class closely follows the proposed implementation of the landmark
    paper "Attention Is All You Need"
    """

    def __init__(self,
                 embed_dim: int,
                 attention_head_count: int,
                 feedforward_dim: int = 2048,
                 dropout: float = 0.1,
                 normalize_before: bool = False):
        """Constructs the base layer of a Transformer Encoder block with
        reasonable default parameters.

        dropout = 0.1 is the default as in the paper.
        """
        super().__init__()

        self.normalize_before = normalize_before

        self.self_attention = torch.nn.MultiheadAttention(
            embed_dim, attention_head_count, dropout=dropout,
            batch_first=True
        )

        self.dropout_1 = torch.nn.Dropout(p=dropout)
        self.layer_norm_1 = torch.nn.LayerNorm(embed_dim)

        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim, attention_head_count, dropout=dropout,
            batch_first=True
        )
        self.dropout_2 = torch.nn.Dropout(p=dropout)
        self.layer_norm_2 = torch.nn.LayerNorm(embed_dim)

        self.feedforward = torch.nn.Sequential(OrderedDict([
            ("tsfm-enc-linear-1", torch.nn.Linear(embed_dim, feedforward_dim)),
            ("tsfm-enc-activation", torch.nn.ReLU()),
            ("tsfm-enc-dropout", torch.nn.Dropout(p=dropout)),
            ("tsfm-enc-linear-2", torch.nn.Linear(feedforward_dim, embed_dim))
        ]))
        self.dropout_3 = torch.nn.Dropout(p=dropout)
        self.layer_norm_3 = torch.nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_embedding: torch.Tensor,
        positional_encoding: torch.Tensor | None,
        output_embedding: torch.Tensor,
        attn_mask: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplemented



class TransformerDecoder(torch.nn.Module):
    """
    The TransformerDecoder class is typically a stack of cross-attention layers.
    It is meant to be used with the TransformerDecoderLayer class.

    We follow closely the implementation proposed by the landmark paper
    "Attention Is All You Need" and other research papers.
    """

    def __init__(self,
                 decoder_layer: torch.nn.Module,
                 num_layers: int = 6,
                 norm: torch.nn.Module | None = None):
        """ Constructs a transformer encoder block.

        By default we construct 6 encoder layers, which has been the default
        parameters in many research papers since the pape "Attention Is All You
        Need".

        Parameters
        ----------

        encoder_layer: torch.nn.Module
            Typically a self-attention encoding layer that we want to replicate.

        num_layers: int
            The number of self-attention encoding layers.

        norm: torch.nn.Module
            Optional layer normalization
        """

        super().__init__()
        self.layers = torch.nn.ModuleList([
            copy.deepcopy(decoder_layer) for _ in range(num_layers)
        ])
        self.norm = norm

    def forward(
        self,
        input_embedding: torch.Tensor,
        positional_encoding: torch.Tensor | None,
        output_embedding: torch.Tensor,
        attn_mask: torch.Tensor
    ) -> torch.Tensor:
        y = output_embedding
        for layer in self.layers:
            y = layer(input_embedding, positional_encoding, y, attn_mask)
        if self.norm is not None:
            y = self.norm(y)
        return y

