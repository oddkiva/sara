import torch

from oddkiva.brahma.torch.transformers.embedding.positional_sine_embedding \
    import (
        PositionalSineEmbedding2D
    )
from oddkiva.brahma.torch.transformers.encoder import (
    TransformerEncoderLayer,
    TransformerEncoder
)


class AIFI(torch.nn.Module):
    """
    AIFI stands for Attention-based Intra-Scale Feature Interaction.

    It is basically a self-attention module that improves the coarsest feature
    map of the feature pyramid produced by a CNN backbone, by capturing
    long-distance relationships between features that CNN backbone could not
    reach.
    """

    def __init__(self,
                 feature_dim: int,
                 attention_head_count: int,
                 feedforward_dim: int = 2048,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 num_layers: int = 6,
                 norm: torch.nn.Module | None = None):
        super().__init__()

        tsfm_enc_layer = TransformerEncoderLayer(feature_dim,
                                                 attention_head_count,
                                                 feedforward_dim=feedforward_dim,
                                                 dropout=dropout,
                                                 normalize_before=normalize_before)

        self.positional_encoding_fn = PositionalSineEmbedding2D(feature_dim)

        self.transformer_encoder = TransformerEncoder(tsfm_enc_layer,
                                                      num_layers,
                                                      norm)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        n, c, h, w = X.shape
        pe = self.positional_encoding_fn((w, h))
        pe_flat = pe.flatten(0, 1)[None, ...]

        # The feature map X has shape (N, C, H, W).
        # Flatten the spatial dimension (N, C, HW)
        # Permute the spatial dimension (N, HW, C)
        X_flat = X.flatten(2).permute(0, 2, 1)
        return self.transformer_encoder.forward(X_flat, pe_flat)
