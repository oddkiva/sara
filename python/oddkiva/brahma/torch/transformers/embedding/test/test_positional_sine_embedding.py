import pytest

import torch

from oddkiva.brahma.torch.transformers.embedding.positional_sine_embedding \
    import PositionalSineEmbedding2D


def RTDETR_sine_embedding_original_implementation(w, h, embed_dim=256, temperature=10000.):
    """ Copy-pasted from RT-DETR.
    """
    grid_w = torch.arange(int(w), dtype=torch.float32)
    grid_h = torch.arange(int(h), dtype=torch.float32)
    # Notice the misleading notation, because of the wrong indexing...
    # It should be "indexing='xy'" instead.
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
    assert embed_dim % 4 == 0, \
        'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)

    out_w = grid_w.flatten()[..., None] @ omega[None]
    out_h = grid_h.flatten()[..., None] @ omega[None]

    return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]


def test_positional_sine_embedding_2d():
    for embed_dim in [0, 1, 2, 3, 5]:
        with pytest.raises(AssertionError):
            phi = PositionalSineEmbedding2D(embed_dim, (3, 3))()

    for embed_dim in [4, 8, 16, 32]:
        phi = PositionalSineEmbedding2D(16, (3, 3))()
        # Transpose the matrix because of the RT-DETR implementation.
        #                      y  x  c
        phi_flat = phi.permute(1, 0, 2).flatten(0, 1)

        phi_true = RTDETR_sine_embedding_original_implementation(
            3, 3, embed_dim=16)

        assert torch.norm(phi_flat - phi_true) < 1e-6
