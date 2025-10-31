# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torch


class PositionalSineEmbedding2D(torch.nn.Module):
    """
    For prototyping and research purposes, we should not overthink it and favor
    this handcrafted embedding before going down the learnable route.
    """

    def __init__(self,
                 embed_dim: int,
                 temperature: int = int(10 ** 4),
                 scale: float = 1.,
                 normalize_coords_01: bool = False):
        super().__init__()

        assert embed_dim > 0 and embed_dim % 4 == 0, \
            "".join([
                "The 2D positional embed dimension must be positive and ",
                "a multiple of 4 by design!"
            ])
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.scale = scale
        self.normalize_coords_01 = normalize_coords_01

        # Precalculate the frequency sequence vector.
        #
        # Let us calculate the geometric ratio. Notice here that, as in
        # RT-DETR's implementation, we don't use the squared version,
        # which is:
        #
        # self.geom_ratio = (1. / temperature) ** (2. / embed_dim_4)
        #
        self.embed_dim_4 = embed_dim // 4
        self.geom_ratio = (1. / temperature) ** (1. / self.embed_dim_4)
        #
        powers = torch.arange(self.embed_dim_4)
        self.frequency_geom_seq = self.geom_ratio ** powers

    @torch.no_grad()
    def forward(self, wh: tuple[int, int]) -> torch.Tensor:
        """
        Calculates the positional encoding map with shape (H, W, C).

        Notice that the indexing order `ij` or `yx` in which the coordinates
        are enumerated.

        if `X` is feature map of shape (N, C, H, W), the flattened feature map
        `X_flat` is such that:

        ```python
        X[n, y, x, c] == X_flat[n, y * w + x, c]
        ```

        This is an important detail to correctly form the query matrix `Q`
        where:
        ```python
        Q = X_flat + positional_encoding.flatten(0, 1).unsqueeze(0)
        ```
        """
        w, h = wh
        w_inverse = 1. / w
        h_inverse = 1. / h
        x_axis = torch.arange(w)
        y_axis = torch.arange(h)
        y, x = torch.meshgrid(y_axis, x_axis, indexing='ij')
        if self.normalize_coords_01:
            x = x * w_inverse
            y = y * h_inverse

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
        sine_cos_embed = torch.cat((
            presine_y_embed.sin(), presine_y_embed.cos(),
            presine_x_embed.sin(), presine_x_embed.cos()
        ), dim=-1)

        return sine_cos_embed
