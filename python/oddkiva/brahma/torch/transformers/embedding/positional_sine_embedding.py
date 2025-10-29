# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torch


@torch.no_grad()
class PositionalSineEmbedding2D(torch.nn.Module):
    """
    For prototyping and research purposes, let us not overthink it, let us just
    favor this before trying the learnable embedding version.
    """

    def __init__(self,
                 embed_dim: int,
                 domain_sizes: tuple[int, int],
                 temperature: int = 10**4):
        super().__init__()
        self.embed_dim = embed_dim

        self.scale = 2 * torch.pi
        self.domain_sizes = domain_sizes
        self.domain_sizes_inverse = 1 / torch.tensor(
            self.domain_sizes, dtype=torch.float32
        )
        self.temperature = temperature

        # Precalculate the frequency sequence vector.
        self._geom_ratio = (1. / temperature) ** (2. / embed_dim)
        self._frequency_ixs = 2 * (torch.arange(embed_dim) // 2)
        self._frequency_geom_seq = self._geom_ratio ** self._frequency_ixs

        # Precalculate the positional embedding map.
        w, h = self.domain_sizes
        x_axis = torch.arange(w)
        y_axis = torch.arange(h)
        x, y = torch.meshgrid(x_axis, y_axis, indexing='xy')
        xy = torch.cat((x, y), dim=-1)
        xy_rescaled = xy * self.scale

        self._pre_sine_cos_embed = xy_rescaled * self._frequency_geom_seq

        self._sine_cos_embed = self._pre_sine_cos_embed

    def forward(self) -> torch.Tensor:
        return self._sine_cos_embed
