import torch
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

        activation_fn = make_activation_func(activation)
        assert activation_fn is not None

        self._layers = nn.Sequential()
        for in_channels, out_channels in zip(in_dim_seq, out_dim_seq):
            self._layers.append(nn.Linear(in_channels, out_channels))
            self._layers.append(activation_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)

    @property
    def layers(self):
        return self._layers

    @property
    def layer_count(self):
        return len(self.layers) // 2



