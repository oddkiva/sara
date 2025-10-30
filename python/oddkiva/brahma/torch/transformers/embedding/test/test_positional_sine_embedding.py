import pytest

import torch

from oddkiva.brahma.torch.transformers.embedding.positional_sine_embedding \
    import PositionalSineEmbedding2D


def RTDETR_sine_embedding_original_implementation(w, h, embed_dim=256, temperature=10000.):
    """ Copy-pasted from RT-DETR.
    """
    grid_w = torch.arange(int(w), dtype=torch.float32)
    grid_h = torch.arange(int(h), dtype=torch.float32)
    # Notice the misleading notation, but we do want the indexing 'ij'! Because
    # tensor in PyTorch are in row-major and indexed as (n, c, y, x).
    #
    # Luckily, all the feature maps have been square since its inception!
    # 😬😬😬
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
    assert embed_dim % 4 == 0, \
        'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)

    out_w = grid_w.flatten()[..., None] @ omega[None]
    out_h = grid_h.flatten()[..., None] @ omega[None]

    return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]


def test_generation_of_positional_sine_embedding_2d():
    for embed_dim in [0, 1, 2, 3, 5]:
        with pytest.raises(AssertionError):
            phi = PositionalSineEmbedding2D(embed_dim)((3, 3))

    for embed_dim in [4, 8, 16, 32]:
        phi = PositionalSineEmbedding2D(embed_dim)((3, 3))
        phi_flat = phi.flatten(0, 1)

        phi_true = RTDETR_sine_embedding_original_implementation(
            3, 3, embed_dim=embed_dim)

        assert torch.norm(phi_flat - phi_true) < 1e-6

def test_additivity_of_positional_sine_embedding_2d():
    N, C, H, W = (8, 16, 3, 3)
    phi_fn = PositionalSineEmbedding2D(C)
    phi = phi_fn((W, H))
    phi_flat = phi.flatten(0, 1)[None, ...]

    ones = torch.ones((N, C, H, W))
    ones_flat = ones.flatten(2).permute((0, 2, 1))

    query = ones_flat + phi_flat
    assert query.shape == (N, H*W, C)

    phi_rep = (query.permute(0, 2, 1).reshape(N, C, H, W) - ones)
    for n in range(N):
        assert torch.norm(phi_rep[n].permute(1, 2, 0) - phi) < 1e-6
