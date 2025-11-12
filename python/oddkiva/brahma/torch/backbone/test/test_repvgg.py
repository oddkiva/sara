import torch

from oddkiva.brahma.torch.backbone.resnet.vanilla import ConvBNA
from oddkiva.brahma.torch.backbone.repvgg import RepVggBaseLayer

def test_repvgg_base_layer():
    hidden_dim = 256
    block = RepVggBaseLayer(hidden_dim, hidden_dim,
                            stride=1,
                            use_identity_connection=False,
                            activation='silu')

    print(block)

    assert len(block.layers) == 2

    assert type(block.layers[0]) is ConvBNA
    assert block.layers[0].layers[0].in_channels == hidden_dim
    assert block.layers[0].layers[0].out_channels == hidden_dim
    assert block.layers[0].layers[0].kernel_size == (3, 3)
    assert block.layers[0].layers[0].stride == (1, 1)

    assert type(block.layers[1]) is ConvBNA
    assert block.layers[1].layers[0].in_channels == hidden_dim
    assert block.layers[1].layers[0].out_channels == hidden_dim
    assert block.layers[1].layers[0].kernel_size == (1, 1)
    assert block.layers[1].layers[0].stride == (1, 1)

    assert type(block.activation) is torch.nn.SiLU
