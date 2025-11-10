from pathlib import Path

import torch
import torch.nn as nn

from oddkiva.brahma.torch.backbone.resnet.vanilla import (
    ConvBNA,
    make_activation_func
)


class UnbiasedConvBNA(ConvBNA):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, id: int, activation: str ='relu'):
        super().__init__(in_channels, out_channels, kernel_size, stride, True,
                         activation, id, bias=False)


class ResidualBottleneckBlock(nn.Module):
    """
    This class implements a variant of fundamental residual block, which is
    used in the ResNet backbone of RT-DETR v2.

    The key differences from the classical residual bottleneck block are as
    follows.

    - Each convolution operation is unbiased.
    - The downsampling of the feature map happens on the second
      `UnbiasedConvBNA` block and not on the first block.
    - The shortcut connection contains an average pooling layer if the stride
      is 2. This pooling is applied first before the convolutional operation.

    The authors have a different vision of the residual bottleneck block and
    the differences are significant enough to justify the writing of a new
    class. This will keep the code simple.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "relu",
        use_shortcut_connection: bool = True,
        inplace_activation: bool = False
    ):
        """
        Constructs the residual bottleneck block variant used in RT-DETR v2.
        """

        super().__init__()
        self.convs = nn.Sequential(
            UnbiasedConvBNA(in_channels, out_channels, 1, 1,
                            0, activation),  # Id
            UnbiasedConvBNA(out_channels, out_channels, 3, stride,
                            1, activation),  # Id
            UnbiasedConvBNA(out_channels, out_channels * (2**2), 1, 1,
                            2, activation)   # Id
        )

        if use_shortcut_connection:
            if stride == 1:
                self.shortcut = UnbiasedConvBNA(in_channels,
                                                out_channels * (2**2),
                                                1, 1,
                                                0, "linear")
            elif stride == 2:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(2, 2, 0, ceil_mode=True),
                    UnbiasedConvBNA(in_channels, out_channels * (2**2),
                                    1, 1,
                                    0, "linear")
                )
            else:
                ValueError("Unsupported: the stride must be 1 or 2!")
        else:
            self.shortcut = None

        # Add the activation layer
        self.activation = make_activation_func(
            activation, inplace=inplace_activation
        )

        self._in_channels = in_channels
        self._out_channels = out_channels

    def forward(self, x):
        if self.shortcut is None:
            out = self.convs(x)
        else:
            out = self.convs.forward(x) + self.shortcut(x)

        if self.activation is not None:
            out = self.activation(out)

        return out

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels


class ResNet50RTDETRV2Variant(nn.Module):
    """This class implements RT-DETR v2 variant form of ResNet-50.

    This variant has some surprising features.

    - None of its convolutional operations contains a learnable bias.
    - At each image level, only the first residual bottleneck block
      incorporates the shortcut connection, contrary to the classical version
      of ResNet-50, where we add shortcuts connections everywhere.

      This proves we can sparingly use shortcut connection for more GFLOP/s
      without hurting the object detection reliability.
    """

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            # Surprising feature: we downsample straight away from the first
            # convolutional operation.
            #
            # No pooling is happening too.
            nn.Sequential(
                UnbiasedConvBNA(3, 32, 3, 2,   # Downsample here
                                0),            # id
                UnbiasedConvBNA(32, 32, 3, 1,
                                1),            # id
                UnbiasedConvBNA(32, 64, 3, 1,
                                1)             # id
            ),
            # P0
            #
            # No downsampling here like in the classical ResNet-50.
            nn.Sequential(
                # Out dim is not 64! But 256 (cf. implementation).
                ResidualBottleneckBlock(64, 64, 1, "relu"),
                # Note that the following residual blocks **do not** use the
                # shortcut connections in RT-DETR v2!
                ResidualBottleneckBlock(256, 64, 1, "relu",
                                        use_shortcut_connection=False),
                ResidualBottleneckBlock(256, 64, 1, "relu",
                                        use_shortcut_connection=False),
            ),
            # P1
            nn.Sequential(
                ResidualBottleneckBlock(256, 128, 2, "relu"),
                # Note that the following residual blocks **do not** use the
                # shortcut connections in RT-DETR v2!
                ResidualBottleneckBlock(512, 128, 1, "relu",
                                        use_shortcut_connection=False),
                ResidualBottleneckBlock(512, 128, 1, "relu",
                                        use_shortcut_connection=False),
                ResidualBottleneckBlock(512, 128, 1, "relu",
                                        use_shortcut_connection=False),
            ),
            # P2
            nn.Sequential(
                ResidualBottleneckBlock(512, 256, 2, "relu"),
                # Note that the following residual blocks **do not** use the
                # shortcut connections in RT-DETR v2!
                ResidualBottleneckBlock(1024, 256, 1, "relu",
                                        use_shortcut_connection=False),
                ResidualBottleneckBlock(1024, 256, 1, "relu",
                                        use_shortcut_connection=False),
                ResidualBottleneckBlock(1024, 256, 1, "relu",
                                        use_shortcut_connection=False),
                ResidualBottleneckBlock(1024, 256, 1, "relu",
                                        use_shortcut_connection=False),
                ResidualBottleneckBlock(1024, 256, 1, "relu",
                                        use_shortcut_connection=False),
            ),
        # P3
        nn.Sequential(
            ResidualBottleneckBlock(1024, 512, 2, "relu"),
            # Note that the following residual blocks **do not** use the
            # shortcut connections in RT-DETR v2!
            ResidualBottleneckBlock(2048, 512, 1, "relu",
                                    use_shortcut_connection=False),
            ResidualBottleneckBlock(2048, 512, 1, "relu",
                                    use_shortcut_connection=False),
        ),
        )

    def forward(self, x):
        return self.blocks.forward(x)


class RTDETRV2Checkpoint:

    res_net_arch_levels = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }

    batch_norm_param_names = ['weight', 'bias', 'running_mean', 'running_var']

    def __init__(self, ckpt_fp: Path, ):
        self.ckpt = torch.load(ckpt_fp)

    @property
    def model_weights(self):
        return self.ckpt['ema']['module']

    @property
    def backbone_keys(self):
        return [k for k in self.model_weights.keys() if 'backbone' in k]

    @property
    def encoder_keys(self):
        return [k for k in self.model_weights.keys() if 'encoder' in k]

    @property
    def decoder_keys(self):
        return [k for k in self.model_weights.keys() if 'decoder' in k]

    @property
    def backbone_weights(self):
        return {k: self.model_weights[k] for k in self.backbone_weights}

    @property
    def encoder_weights(self):
        return {k: self.model_weights[k] for k in self.encoder_weights}

    @property
    def decooder_weights(self):
        return {k: self.model_weights[k] for k in self.decoder_weights}

    def conv1_key(self, subblock_idx: int):
        return f'backbone.conv1.conv1_{subblock_idx}'

    def conv1_conv_key(self, subblock_idx: int):
        return f"{self.conv1_key(subblock_idx)}.conv"

    def conv1_bn_key(self, subblock_idx: int):
        return f"{self.conv1_key(subblock_idx)}.norm"

    def conv1_conv_weight(self, subblock_idx: int) -> torch.Tensor:
        key = f"{self.conv1_conv_key(subblock_idx)}.weight"
        return self.model_weights[key]

    def conv1_bn_weights(
        self, subblock_idx: int
    ) -> dict[str, torch.Tensor]:
        keys = {
            param: f"{self.conv1_bn_key(subblock_idx)}.{param}"
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }
        return {
            param: self.model_weights[keys[param]]
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }

    def bottleneck_key(self, block_idx: int, subblock_idx: int) -> str:
        return f'backbone.res_layers.{block_idx}.blocks.{subblock_idx}'

    def bottleneck_branch_key(self, block_idx: int, subblock_idx: int,
                              branch: str) -> str:
        bottleneck_key = self.bottleneck_key(block_idx, subblock_idx)
        return f'{bottleneck_key}.branch2{branch}'

    def bottleneck_branch_conv_key(
        self, block_idx: int, subblock_idx: int, branch: str
    ) -> str:
        bottleneck_key = self.bottleneck_key(block_idx, subblock_idx)
        return f'{bottleneck_key}.branch2{branch}.conv'

    def bottleneck_branch_conv_weight(
        self, block: int, subblock: int, branch: str
    ) -> torch.Tensor:
        parent_key = self.bottleneck_branch_conv_key(block, subblock, branch)
        key = f"{parent_key}.weight"
        return self.model_weights[key]

    def bottleneck_branch_bn_key(
        self, block_idx: int, subblock_idx: int, branch: str
    ) -> str:
        bottleneck_key = self.bottleneck_key(block_idx, subblock_idx)
        return f'{bottleneck_key}.branch2{branch}.norm'

    def bottleneck_branch_bn_weights(
        self, block: int, subblock: int, branch: str
    ) -> dict[str, torch.Tensor]:
        parent_key = self.bottleneck_branch_bn_key(block, subblock, branch)
        keys = {
            param: f"{parent_key}.{param}"
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }
        return {
            param: self.model_weights[keys[param]]
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }

    def bottleneck_short_key(self, block_idx: int, subblock_idx: int) -> str:
        bottleneck_key = self.bottleneck_key(block_idx, subblock_idx)
        return f'{bottleneck_key}.short'

    def bottleneck_short_conv_key(self, block_idx: int, subblock_idx: int) -> str:
        bottleneck_key = self.bottleneck_key(block_idx, subblock_idx)
        if block_idx == 0:
            return f'{bottleneck_key}.short.conv'
        else:
            return f'{bottleneck_key}.short.conv.conv'

    def bottleneck_short_bn_key(self, block_idx: int, subblock_idx: int) -> str:
        bottleneck_key = self.bottleneck_key(block_idx, subblock_idx)
        if block_idx == 0:
            return f'{bottleneck_key}.short.norm'
        else:
            return f'{bottleneck_key}.short.conv.norm'

    def bottleneck_short_conv_weight(
        self, block: int, subblock: int
    ) -> torch.Tensor:
        parent_key = self.bottleneck_short_conv_key(block, subblock)
        key = f"{parent_key}.weight"
        return self.model_weights[key]

    def bottleneck_short_bn_weights(
        self, block: int, subblock: int
    ) -> dict[str, torch.Tensor]:
        parent_key = self.bottleneck_short_bn_key(block, subblock)
        keys = {
            param: f"{parent_key}.{param}"
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }
        return {
            param: self.model_weights[keys[param]]
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }

    def _copy_conv_bna_weights(self, my_block: UnbiasedConvBNA,
                               conv_weight: torch.Tensor,
                               bn_weights: dict[str, torch.Tensor]) -> None:
        my_conv: nn.Conv2d = my_block.layers[0]
        my_bn: nn.BatchNorm2d = my_block.layers[1]

        assert my_conv.weight.shape == conv_weight.shape
        assert my_bn.weight.shape == bn_weights['weight'].shape
        assert my_bn.bias.shape == bn_weights['bias'].shape
        assert my_bn.running_mean.shape == bn_weights['running_mean'].shape
        assert my_bn.running_var.shape == bn_weights['running_var'].shape

        my_conv.weight.data.copy_(conv_weight)
        my_bn.weight.data.copy_(bn_weights['weight'])
        my_bn.bias.data.copy_(bn_weights['bias'])
        my_bn.running_mean.data.copy_(bn_weights['running_mean'])
        my_bn.running_var.data.copy_(bn_weights['running_var'])

    def _load_backbone_conv_1(self, model):
        for i in range(3):
            my_conv_bna = model.blocks[0][i]

            conv_weight = self.conv1_conv_weight(i + 1)
            bn_weights = self.conv1_bn_weights(i + 1)
            self._copy_conv_bna_weights(my_conv_bna, conv_weight, bn_weights)

    def load_backbone(self):
        model = ResNet50RTDETRV2Variant()

        self._load_backbone_conv_1(model)

        return model


