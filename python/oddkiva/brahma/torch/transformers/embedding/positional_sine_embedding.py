import torch


class PositionalSineEmbedding2D(torch.nn.Module):
    """
    For prototyping and research purposes, let us not overthink it, let us just
    favor this before trying the learnable embedding version.
    """

    def __init__(self,
                 key_dimension: int,
                 domain_sizes: tuple[int, int],
                 temperature: int = 10**4):
        super().__init__()
        self.key_dimension = key_dimension
        self.domain_sizes = domain_sizes
        self.scale = 2 * torch.pi
        self.temperature = temperature

    def forward(self) -> torch.Tensor:
        w, h = self.domain_sizes
        x_axis = torch.arange(w, dtype=torch.float32)
        y_axis = torch.arange(h, dtype=torch.float32)
        x, y = torch.meshgrid(x_axis, y_axis, indexing='xy')
        xy = torch.cat((x, y), dim=-1)
        xy_normalized = xy / torch.tensor(self.domain_sizes)

        # Each tensor is of shape (N, H, W, C)
        # cumsum(1) -> cumsum(axis=1)  # axis 1 --> H --> y dimension
        # cumsum(2) -> cumsum(axis=2)  # axis 2 --> W --> x dimension
        # If the mask is True everywhere, then it is like.
        # not_mask.cumsum(1) is equivalent to torch.arange(H).
        # cumsum(1) is useful if the mask has holes.
        #
        # [True, True, True, False, False, True]
        # [   0,    1,    2,     2,     2,    3]
        #                        ^      ^
        #                        |      |
        # [   0,    1,    2,     |,     |,    3]
        #                        |      |
        #                        --------
        #                        |
        #                        we don't care
        y_embed = x_axis
        x_embed = y_axis
        if self.normalize:
            w, h = self.domain_sizes
            x_embed = x_embed / w * self.scale
            y_embed = y_embed / h * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed / dim_t
        pos_y = y_embed / dim_t
        pos_x = torch.stack(
            (pos_x[0::2].sin(), pos_x[1::2].cos()),
            dim=-1
        ).flatten(-1)
        pos_y = torch.stack(
            (pos_y[0::2].sin(), pos_y[1::2].cos()),
            dim=-1
        ).flatten(-1)
        pos = torch.cat((pos_y, pos_x), dim=-1)
        return pos
