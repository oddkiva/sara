from pathlib import Path
import pytest

import torch

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.backbone.resnet50 import ResNet50RTDETRV2Variant


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')


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



def test_rtdetrv2_resnet50_backbone_reconstruction():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH)

    model = ResNet50RTDETRV2Variant()

    for i in [1, 2, 3]:
        my_conv = model.blocks[0][i - 1].layers[0]
        my_bn = model.blocks[0][i - 1].layers[1]

        conv_weight = ckpt.conv1_conv_weight(i)
        bn_weights = ckpt.conv1_bn_weights(i)

        assert my_conv.weight.shape == conv_weight.shape
        assert my_bn.weight.shape == bn_weights['weight'].shape
        assert my_bn.bias.shape == bn_weights['bias'].shape
        assert my_bn.running_mean.shape == bn_weights['running_mean'].shape
        assert my_bn.running_var.shape == bn_weights['running_var'].shape

    arch_levels = RTDETRV2Checkpoint.res_net_arch_levels[50]
    print(arch_levels)


    indexing = {0: 'a', 1: 'b', 2: 'c'}
    for block_idx, subblock_count in enumerate(arch_levels):

        for i in range(subblock_count):
            # ckpt.bottleneck_branch_conv_weight(block_idx, i, 'a')
            # ckpt.bottleneck_branch_conv_weight(block_idx, i, 'b')
            # ckpt.bottleneck_branch_conv_weight(block_idx, i, 'c')
            # ckpt.bottleneck_branch_bn_weights(block_idx, i, 'a')
            # ckpt.bottleneck_branch_bn_weights(block_idx, i, 'b')
            # ckpt.bottleneck_branch_bn_weights(block_idx, i, 'c')

            my_bottleneck_stack = model.blocks[block_idx + 1]
            my_bottleneck_block = my_bottleneck_stack[i]

            # Loop through ['branch2a', 'branch2b', 'branch2c']
            for j in range(3):
                my_conv = my_bottleneck_block.convs[j].layers[0]
                my_bn = my_bottleneck_block.convs[j].layers[1]

                letter = indexing[j]
                conv_weight = ckpt.bottleneck_branch_conv_weight(block_idx, i, letter)
                bn_weights = ckpt.bottleneck_branch_bn_weights(block_idx, i, letter)

                assert my_conv.weight.shape == conv_weight.shape
                assert my_bn.weight.shape == bn_weights['weight'].shape
                assert my_bn.bias.shape == bn_weights['bias'].shape
                assert my_bn.running_mean.shape == bn_weights['running_mean'].shape
                assert my_bn.running_var.shape == bn_weights['running_var'].shape

            if i == 0:
                my_shortcut = my_bottleneck_block.shortcut
                my_conv = my_shortcut.layers[0]
                my_bn = my_shortcut.layers[1]

                conv_weight = ckpt.bottleneck_short_conv_weight(block_idx, i)
                bn_weights = ckpt.bottleneck_short_bn_weights(block_idx, i)

                assert my_conv.weight.shape == conv_weight.shape
                assert my_bn.weight.shape == bn_weights['weight'].shape
                assert my_bn.bias.shape == bn_weights['bias'].shape
                assert my_bn.running_mean.shape == bn_weights['running_mean'].shape
                assert my_bn.running_var.shape == bn_weights['running_var'].shape
            else:
                my_shortcut = my_bottleneck_block.shortcut
                assert my_shortcut is None

                with pytest.raises(KeyError):
                    ckpt.bottleneck_short_conv_weight(block_idx, i)
                    ckpt.bottleneck_short_bn_weights(block_idx, i)
