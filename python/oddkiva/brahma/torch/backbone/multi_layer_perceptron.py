# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torch.nn as nn

from oddkiva.brahma.torch.backbone.resnet.vanilla import make_activation_func


class MultiLayerPerceptron(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 layer_count: int,
                 activation: str ='relu'):
        super().__init__()

        hidden_dim_seq = [hidden_dim] * (layer_count - 1)
        in_dim_seq = [in_dim] + hidden_dim_seq
        out_dim_seq = hidden_dim_seq + [out_dim]

        self.layers = nn.ModuleList(
            nn.Linear(in_channels, out_channels)
            for in_channels, out_channels in zip(in_dim_seq, out_dim_seq)
        )
        self.activation = make_activation_func(activation)
        assert self.activation is not None


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if i < self.layer_count - 1 else layer(x)
        return x


    @property
    def layer_count(self) -> int:
        return len(self.layers)
