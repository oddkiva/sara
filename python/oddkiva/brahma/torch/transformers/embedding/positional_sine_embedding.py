# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torch


class PositionalSineEmbedding2D(torch.nn.Module):
    """
    For prototyping and research purposes, let us not overthink it, let us just
    favor this before trying the learnable embedding version.
    """

    def __init__(self,
                 embed_dim: int,
                 domain_sizes: tuple[int, int],
                 temperature: int = int(10 ** 4),
                 scale: float = 1.,
                 normalize_geometric_initial_values: bool = False):
        super().__init__()

        assert embed_dim > 0 and embed_dim % 4 == 0, \
            "".join([
                "The 2D positional embed dimension must be positive and ",
                "a multiple of 4 by design!"
            ])
        self.scale = scale
        self.domain_sizes = domain_sizes
        self.temperature = temperature

        # Precalculate the frequency sequence vector.
        #
        # Let us calculate the geometric ratio. Notice here that, as in
        # RT-DETR's implementation, we don't use the squared version,
        # which is:
        #
        # self.geom_ratio = (1. / temperature) ** (2. / embed_dim_4)
        #
        embed_dim_4 = embed_dim // 4
        self.geom_ratio = (1. / temperature) ** (1. / embed_dim_4)
        #
        powers = torch.arange(embed_dim_4)
        self.frequency_geom_seq = self.geom_ratio ** powers

        # Precalculate the positional embedding map.
        w, h = domain_sizes
        x_axis = torch.arange(w)
        y_axis = torch.arange(h)
        x, y = torch.meshgrid(x_axis, y_axis, indexing='xy')
        if normalize_geometric_initial_values:
            x = x / w
            y = y / h

        x_rescaled = x * self.scale
        y_rescaled = y * self.scale

        # I don't like the flatten operation in RT-DETR.
        presine_x_embed \
            = x_rescaled[..., None] \
            * self.frequency_geom_seq[None, None, :]
        presine_y_embed \
            = y_rescaled[..., None] \
            * self.frequency_geom_seq[None, None, :]

        # Let's not recompose the positional embedding as explained in the
        # paper "Attention is All You Need", the order of the components does
        # not matter.
        #
        # We can concatenate the 4 vectors:
        self.sine_cos_embed = torch.cat((
            presine_x_embed.sin(), presine_x_embed.cos(),
            presine_y_embed.sin(), presine_y_embed.cos()
        ), dim=-1)

    @torch.no_grad()
    def forward(self) -> torch.Tensor:
        return self.sine_cos_embed
