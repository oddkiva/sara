# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import copy
from collections import OrderedDict

import torch


class TransformerEncoderLayer(torch.nn.Module):
    """This class closely follows the proposed implementation of the landmark
    paper "Attention Is All You Need"
    """

    def __init__(self, embed_dim: int,
                 num_heads: int,
                 feedforward_dim: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 normalize_before: bool = Fale):
        """Constructs the base layer of a Transformer Encoder block with
        reasonable default parameters.

        dropout = 0.1 is the default as in the paper.
        """
        super().__init__()

        self.normalize_before = normalize_before

        self.self_attention = torch.nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout,
            batch_first=True
        )

        self.dropout_1 = torch.nn.Dropout(p=dropout)
        self.layer_norm_1 = torch.nn.LayerNorm(embed_dim)

        self.feedforward = torch.nn.Sequential(OrderedDict([
            ("tsfm-enc-linear-1", torch.nn.Linear(embed_dim, feedforward_dim)),
            ("tsfm-enc-activation", torch.nn.ReLU()),
            ("tsfm-enc-dropout", torch.nn.Dropout(p=dropout)),
            ("tsfm-enc-linear-2", torch.nn.Linear(feedforward_dim, embed_dim))
        ]))

        self.dropout_2 = torch.nn.Dropout(p=dropout)
        self.layer_norm_2 = torch.nn.LayerNorm(embed_dim)

    def forward_and_prenormalize(
        self,
        input_embedding: torch.Tensor,
        positional_encoding: torch.Tensor,
        attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # 1. Normalize -> Self-Attention -> Add
        input_embedding_normalized = self.layer_norm_1(input_embedding)
        queries = input_embedding_normalized + positional_encoding
        keys = queries
        values = input_embedding_normalized
        enhanced_value_residuals, _ = self.self_attention(
            queries, keys, values,
            attn_mask=attn_mask
        )
        # Perturb the enhanced value residuals to avoid overfitting in the
        # self-attention block.
        enhanced_value_residuals = self.dropout1(enhanced_value_residuals)
        enhanced_values = values + enhanced_value_residuals

        # 2. Normalize -> FFN -> Add
        #
        enhanced_values = self.layer_norm_2(enhanced_values)
        enhanced_value_residuals = self.feed_forward(enhanced_values)
        # Perturb the enhanced value residuals to avoid overfitting in the
        # feed-forward block.
        enhanced_value_residuals = self.dropout_2(enhanced_values)
        # Now apply the Add layer.
        enhanced_values = enhanced_values + enhanced_value_residuals

        return enhanced_values

    def forward_and_postnormalize(
        self,
        input_embedding: torch.Tensor,
        positional_encoding: torch.Tensor,
        attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # 1. Self-Attention -> Add+Norm
        queries = input_embedding + positional_encoding
        keys = queries
        values = input_embedding
        enhanced_value_residuals, _ = self.self_attention.forward(
            queries, keys, values,
            attn_mask=attn_mask
        )
        # Perturb the enhanced value residuals to avoid overfitting in the
        # self-attention block.
        enhanced_value_residuals = self.dropout1(enhanced_value_residuals)
        # Now apply the Add+Norm layer.
        enhanced_values = self.layer_norm_1(
            values + enhanced_value_residuals
        )

        # 2. FFN -> Add+Norm
        enhanced_value_residuals = self.feedforward(enhanced_values)
        # Perturb the enhanced value residuals to avoid overfitting in the
        # feed-forward block.
        enhanced_value_residuals = self.dropout_2(enhanced_values)
        # Now apply the Add+Norm layer.
        enhanced_values = self.layer_norm_2(
            enhanced_values + enhanced_value_residuals
        )

        return enhanced_values

    def forward(
        self,
        input_embedding: torch.Tensor,
        positional_encoding: torch.Tensor,
        attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.normalize_before:
            return self.forward_and_prenormalize(input_embedding,
                                                 positional_encoding,
                                                 attn_mask)
        return self.forward_and_postnormalize(input_embedding,
                                              positional_encoding,
                                              attn_mask)


class TransformerEncoder(torch.nn.Module):
    """
    The TransformerEncoder class is typically a stack of self-attention layers.
    It is meant to be used with the TransformerEncoderLayer class.

    We follow closely the implementation proposed by the landmark paper
    "Attention Is All You Need" and other research papers.
    """

    def __init__(self,
                 encoder_layer: torch.nn.Module,
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
            copy.deepcopy(encoder_layer) for _ in range(num_layers)
        ])
        self.norm = norm

    def forward(
        self,
        features: torch.Tensor,
        positional_encoding: torch.Tensor | None,
        attn_mask: torch.Tensor
    ) -> torch.Tensor:
        y = features
        for layer in self.layers:
            y = layer(y, positional_encoding, attn_mask)
        if self.norm is not None:
            y = self.norm(y)
        return y
