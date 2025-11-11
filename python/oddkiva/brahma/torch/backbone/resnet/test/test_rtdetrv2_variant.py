import pytest

import torch

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.backbone.resnet.rtdetrv2_variant import (
    ConvBNA,
    ResNet50RTDETRV2Variant,
    RTDETRV2Checkpoint
)


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')

DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2-data.pt')


def test_rtdetrv2_resnet50_variant_construction():
    data = torch.load(DATA_FILEPATH)

    model = ResNet50RTDETRV2Variant()
    model.freeze_batch_norm(model)

    x = data['input']
    y = model.blocks[0](x)
    y = model.blocks[1](y)
    y = model.blocks[2](y)
    y = model.blocks[3](y)
    y = model.blocks[4](y)

    y = model(x)


def test_rtdetrv2_resnet50_backbone_param_shape():
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

        # TODO: start loading the weights here and check from here.


    arch_levels = RTDETRV2Checkpoint.res_net_arch_levels[50]
    print(arch_levels)


    indexing = {0: 'a', 1: 'b', 2: 'c'}
    for block_idx, subblock_count in enumerate(arch_levels):

        for i in range(subblock_count):
            print(f"block_idx = {block_idx}  i = {i}")
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
                if type(my_shortcut) is ConvBNA:
                    my_convbna = my_shortcut
                elif type(my_shortcut) is torch.nn.Sequential:
                    my_convbna = my_shortcut[1]
                else:
                    TypeError("This should not happen")
                my_conv = my_convbna.layers[0]
                my_bn = my_convbna.layers[1]

                # TODO: copy the weight and bias.

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


def test_rtdetrv2_resnet_backbone_weight_loading():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))

    model = ckpt.load_backbone()
