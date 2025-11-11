import pytest
from loguru import logger
from pathlib import Path

import torch

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.backbone.resnet.rtdetrv2_variant import (
    UnbiasedConvBNA,
    ResNet50RTDETRV2Variant,
    RTDETRV2Checkpoint
)


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')

# DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
#                  'rtdetrv2-data.pt')

DATA_FILEPATH = Path('/home/david/Desktop/rtdetrv2-data.pt')

# def test_rtdetrv2_resnet50_variant_construction():
#     data = torch.load(DATA_FILEPATH)
#
#     model = ResNet50RTDETRV2Variant()
#     model.freeze_batch_norm(model)
#
#     x = data['input']
#     y = model.blocks[0](x)
#     y = model.blocks[1](y)
#     y = model.blocks[2](y)
#     y = model.blocks[3](y)
#     y = model.blocks[4](y)
#
#     y = model(x)
#
#
# def test_rtdetrv2_resnet50_backbone_param_shape():
#     ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH)
#
#     model = ResNet50RTDETRV2Variant()
#
#     for i in [1, 2, 3]:
#         my_conv = model.blocks[0][i - 1].layers[0]
#         my_bn = model.blocks[0][i - 1].layers[1]
#
#         conv_weight = ckpt.conv1_conv_weight(i)
#         bn_weights = ckpt.conv1_bn_weights(i)
#
#         assert my_conv.weight.shape == conv_weight.shape
#         assert my_bn.weight.shape == bn_weights['weight'].shape
#         assert my_bn.bias.shape == bn_weights['bias'].shape
#         assert my_bn.running_mean.shape == bn_weights['running_mean'].shape
#         assert my_bn.running_var.shape == bn_weights['running_var'].shape
#
#         # TODO: start loading the weights here and check from here.
#
#
#     arch_levels = RTDETRV2Checkpoint.resnet_arch_levels[50]
#     print(arch_levels)
#
#
#     indexing = {0: 'a', 1: 'b', 2: 'c'}
#     for block_idx, subblock_count in enumerate(arch_levels):
#
#         for i in range(subblock_count):
#             print(f"block_idx = {block_idx}  i = {i}")
#             my_bottleneck_stack = model.blocks[block_idx + 1]
#             my_bottleneck_block = my_bottleneck_stack[i]
#
#             # Loop through ['branch2a', 'branch2b', 'branch2c']
#             for j in range(3):
#                 my_conv = my_bottleneck_block.convs[j].layers[0]
#                 my_bn = my_bottleneck_block.convs[j].layers[1]
#
#                 letter = indexing[j]
#                 conv_weight = ckpt.bottleneck_branch_conv_weight(block_idx, i, letter)
#                 bn_weights = ckpt.bottleneck_branch_bn_weights(block_idx, i, letter)
#
#                 assert my_conv.weight.shape == conv_weight.shape
#                 assert my_bn.weight.shape == bn_weights['weight'].shape
#                 assert my_bn.bias.shape == bn_weights['bias'].shape
#                 assert my_bn.running_mean.shape == bn_weights['running_mean'].shape
#                 assert my_bn.running_var.shape == bn_weights['running_var'].shape
#
#             if i == 0:
#                 my_shortcut = my_bottleneck_block.shortcut
#                 if type(my_shortcut) is UnbiasedConvBNA:
#                     my_convbna = my_shortcut
#                 elif type(my_shortcut) is torch.nn.Sequential:
#                     my_convbna = my_shortcut[1]
#                 else:
#                     TypeError("This should not happen")
#                 my_conv = my_convbna.layers[0]
#                 my_bn = my_convbna.layers[1]
#
#                 # TODO: copy the weight and bias.
#
#                 conv_weight = ckpt.bottleneck_short_conv_weight(block_idx, i)
#                 bn_weights = ckpt.bottleneck_short_bn_weights(block_idx, i)
#
#                 assert my_conv.weight.shape == conv_weight.shape
#                 assert my_bn.weight.shape == bn_weights['weight'].shape
#                 assert my_bn.bias.shape == bn_weights['bias'].shape
#                 assert my_bn.running_mean.shape == bn_weights['running_mean'].shape
#                 assert my_bn.running_var.shape == bn_weights['running_var'].shape
#             else:
#                 my_shortcut = my_bottleneck_block.shortcut
#                 assert my_shortcut is None
#
#                 with pytest.raises(KeyError):
#                     ckpt.bottleneck_short_conv_weight(block_idx, i)
#                     ckpt.bottleneck_short_bn_weights(block_idx, i)


def test_rtdetrv2_resnet_backbone_weight_loading():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))
    model = ckpt.load_backbone()

    x = data['input']
    intermediate_outs = data['intermediate']

    conv1 = model.blocks[0]
    with torch.no_grad():
        conv1_x = conv1(x)
        conv1_x_true = intermediate_outs['conv1']
        assert torch.norm(conv1_x - conv1_x_true) < 10e-30

    b00a_true = intermediate_outs['debug']['block00a']
    b00b_true = intermediate_outs['debug']['block00b']
    b00c_true = intermediate_outs['debug']['block00c']
    b00s_true = intermediate_outs['debug']['block00s']

    bblock00 = model.blocks[1][0]
    bblock00a = model.blocks[1][0].convs[0]
    bblock00b = model.blocks[1][0].convs[1]
    bblock00c = model.blocks[1][0].convs[2]
    bblock00s = model.blocks[1][0].shortcut

    bblock01 = model.blocks[1][1]
    bblock01a = model.blocks[1][1].convs[0]
    bblock01b = model.blocks[1][1].convs[1]
    bblock01c = model.blocks[1][1].convs[2]

    bblock02 = model.blocks[1][2]

    # Check the implementation of the bottleneck block (0, 0).
    with torch.no_grad():
        b00a = bblock00a(conv1_x)
        b00b = bblock00b(b00a)
        b00c = bblock00c(b00b)
        b00s = bblock00s(conv1_x)
        assert torch.norm(b00a - b00a_true) < 1e-12
        assert torch.norm(b00b - b00b_true) < 1e-12
        assert torch.norm(b00c - b00c_true) < 1e-12
        assert torch.norm(b00s - b00s_true) < 1e-12

    # Check the implementation of the bottleneck block (0, 1).
    b01a_true = intermediate_outs['debug']['block01a']
    b01b_true = intermediate_outs['debug']['block01b']
    b01c_true = intermediate_outs['debug']['block01c']
    with torch.no_grad():
        b00 = bblock00(conv1_x)

        b01a = bblock01a(b00)
        b01b = bblock01b(b01a)
        b01c = bblock01c(b01b)
        assert torch.norm(b01a - b01a_true) < 1e-12
        assert torch.norm(b01b - b01b_true) < 1e-12
        assert torch.norm(b01c - b01c_true) < 1e-12

    # Check the implementation of a bottleneck stack.
    b00_true = intermediate_outs['debug']['bottleneck00']
    b01_true = intermediate_outs['debug']['bottleneck01']
    # b02_true = intermediate_outs['debug']['bottleneck02']

    with torch.no_grad():
        b00 = bblock00(conv1_x)
        b01 = bblock01(b00)
        b02 = bblock02(b01)
        assert torch.norm(b00 - b00_true) < 1e-12
        assert torch.norm(b01 - b01_true) < 1e-12
        # assert torch.norm(b02 - b02_true) < 1e-12

    # res = [model.blocks[i] for i in range(1, len(model.blocks))]
    # with torch.no_grad():
    #     res_true = intermediate_outs['res_layers']

    #     res0 = model.blocks[1](conv1_x)
    #     res1 = model.blocks[2](res0)
    #     res2 = model.blocks[3](res1)
    #     res3 = model.blocks[4](res2)

    #     res = [res0, res1, res2, res3]
    #     for i, (res_i, res_i_true) in enumerate(zip(res, res_true)):
    #         diff = torch.norm(res_i - res_i_true)
    #         logger.info(f"[{i}] diff = {diff}")
    #         # assert diff < 1e-12
