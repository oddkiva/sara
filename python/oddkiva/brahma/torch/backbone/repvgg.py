import torch

from oddkiva.brahma.torch.backbone.resnet.rtdetrv2_variant import (
    ConvBNA,
    UnbiasedConvBNA,
    make_activation_func
)


class RepVggBaseLayer(torch.nn.Module):
    r"""This class implements the architecture proposed in the paper
    [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/pdf/2101.03697)

    The details are summarized in Figure 2 and Figure 4.
    Notice that in Figure 4, batch normalization also takes place.

    RepVgg has two dually equivalent architectures:

    - At training time, like ResNet, it has two skip connections that avoids
      vanishing gradients.

    - At inference time, the three branches are equivalently merged into a
      single convolutional block, which makes it computationally efficient when
      deploying in production environments.

    One major technical detail of this block, that we should pay attention to,
    is that the convolutional operations in each branch are *unbiased*, i.e.,

    $$
    \mathrm{conv}(\mathbf{x}) = \mathbf{W} * \mathbf{x}
    $$

    By zeroing the bias vector $\mathbf{b}$, we equivalently collapse the 3
    convolutional branches into a single convolutional operation at inference
    time. Please refer to equations (1-4) of the paper.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 2,
                 use_identity_connection: bool = False,
                 activation: str = 'relu',
                 inplace_activation: bool = False):
        r"""Constructs a RegVggBaseLayer.

        The current parameters should be enough to accommodate:

        - the different types of RepVGG layers as exposed in the original
          paper, and
        - the specific needs of RT-DETR, which does not do any downsampling at
          all.

        In any case, there are at least two branches
        ($\mathrm{conv}_{3\times3}$, $\mathrm{conv}_{1\times1}$) and an
        optional identity branch:

        $$
        \mathbf{y} = \mathrm{Conv}_{3 \times 3}(\mathbf{x})
                   + \mathrm{Conv}_{1 \times 1}(\mathbf{x})
                   + \mathbf{x}
        $$

        parameters:
            in_channels: the input feature dimension
            out_channels: the output feature dimension
            stride: the spatial step size for downsampling purposes
            use_identity_connection: optionally add the identity connection
            activation: the activation function
            inplace_activation: apply the activation function in-place
        """
        super().__init__()

        self.use_identity_connection = use_identity_connection

        if use_identity_connection:
            self.layers = torch.nn.ModuleList([
                UnbiasedConvBNA(in_channels, out_channels, 3, stride, 3,
                                activation=None),
                UnbiasedConvBNA(in_channels, out_channels, 1, stride, 1,
                                activation=None),
                torch.nn.BatchNorm2d(out_channels)  # (Identity -> BN2d)
            ])
        else:
            self.layers = torch.nn.ModuleList([
                UnbiasedConvBNA(in_channels, out_channels, 3, stride, 3,  # id
                                activation=None),
                UnbiasedConvBNA(in_channels, out_channels, 1, stride, 1,  # id
                                activation=None)
            ])
        self.activation = make_activation_func(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers[0](x)
        for layer in self.layers[1:]:
            y = y + layer(x)
        if self.activation is None:
            return y
        return self.activation(y)

    # def deploy_for_inference(self):
    #     if not hasattr(self, 'conv'):
    #         self.conv = torch.nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

    #     kernel, bias = self._compute_equivalent_single_convolution()
    #     self.conv.weight.data = kernel
    #     self.conv.bias.data = bias

    def _compute_equivalent_single_convolution(
        self
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: check the equations (1-4) of the paper
        https://arxiv.org/pdf/2101.03697
        """
        w3x3, b3x3 = self._fuse_bn_tensor(self.layers[0])

        w1x1, b1x1 = self._fuse_bn_tensor(self.layers[1])
        w1x1_as_3x3 = self._transform_w1x1_as_eq_w3x3(w1x1)

        if self.use_identity_connection:
            one1x1 = torch.tensor([[1]])
            one1x1_as_3x3 = self._transform_w1x1_as_eq_w3x3(one1x1)
            return (w3x3 + w1x1_as_3x3 + one1x1_as_3x3,
                    b3x3 + b1x1)
        else:
            return (w3x3 + w1x1_as_3x3,
                    b3x3 + b1x1)

    def _transform_w1x1_as_eq_w3x3(
        self,
        kernel1x1: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_conv_bn_layer(self, branch: UnbiasedConvBNA):
        """
        This is the same implementation as in:
        `oddkiva/shakti/inference/darknet/network.py`
        Track the variable `fuse_conv_bn_layer` in the implementation.
        """
        conv = branch.layers[0]
        kernel = conv.weight

        bn = branch.layers[1]
        eps = bn.eps
        running_mean = bn.running_mean
        running_var = bn.running_var

        gamma = bn.weight
        beta = bn.bias
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        return kernel * t, beta - running_mean * gamma / std


class RepVggStack(torch.nn.Module):
    """
    The `RepVggStack` is a convenience class that implements a sequence of
    repeated base layers `RepVggBaseLayer`, just like the
    `ResidualBottleneckBlock` class is the convenience building block for
    ResNet-50.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 layer_count: int = 4,
                 activation: str = 'silu',
                 inplace_activation: bool = False,
                 use_identity_connection: bool = False):
        """
        Constructs a RepVggStack.

        Most of the default parameters are the default ones used to construct
        the Path-Aggregated Feature Pyramid Network (PA-FPN) of RT-DETR.

        parameters:
            in_channels: the input feature dimension
            out_channels: the output feature dimension
            layer_count: the number of `RepVGGBaseLayer` layers to stack
            activation: the activation function
            inplace_activation: apply the activation function in-place
        """
        super().__init__()
        self.layers = torch.nn.Sequential(*[
            RepVggBaseLayer(in_channels, out_channels, stride=1,
                            use_identity_connection=use_identity_connection,
                            activation=activation,
                            inplace_activation=inplace_activation)
            for _ in range(layer_count)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
