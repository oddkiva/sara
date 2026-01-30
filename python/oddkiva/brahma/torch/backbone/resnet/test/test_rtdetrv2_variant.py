# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from loguru import logger

import torch

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.utils.freeze import freeze_batch_norm
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.checkpoint import (
        ResNet50RTDETRV2Variant,
        RTDETRV2Checkpoint
    )
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.config import RTDETRConfig


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')
DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.data.pt')


def test_rtdetrv2_resnet50_inplace_activation():
    model = ResNet50RTDETRV2Variant()

    def list_activation_layers(module: torch.nn.Module,
                               activations: list[torch.nn.Module]):
        # DFS visit.
        for child_tree_name, child_tree in module.named_children():
            if 'activation' in child_tree_name:
                # the child tree is a node.
                activations.append((child_tree_name, child_tree))
            else:
                list_activation_layers(child_tree, activations)

    activations = []
    list_activation_layers(model, activations)

    assert all([module.inplace for _, module in activations])


def test_rtdetrv2_resnet50_variant_construction():
    data = torch.load(DATA_FILEPATH)

    model = ResNet50RTDETRV2Variant()

    assert model.feature_pyramid_dims == [256, 512, 1024, 2048]

    x = data['input']
    y = model.blocks[0](x)
    y = model.blocks[1](y)
    y = model.blocks[2](y)
    y = model.blocks[3](y)
    y = model.blocks[4](y)

    y = model(x)


def test_rtdetrv2_resnet_backbone_computation_details():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    model = ResNet50RTDETRV2Variant()
    ckpt.load_backbone(model)
    model = freeze_batch_norm(model)

    x = data['input']
    intermediate_outs = data['intermediate']
    backbone_debug_outs = intermediate_outs['backbone']['debug']

    conv1 = model.blocks[0]
    with torch.no_grad():
        conv1_x = conv1(x)
        conv1_x_true = intermediate_outs['conv1']
        assert torch.dist(conv1_x, conv1_x_true) < 10e-30

    # Check the implementation of the bottleneck block (0, 0).
    #
    # 1. Get the blocks
    bblock00 = model.blocks[1][0]
    bblock00a = model.blocks[1][0].convs[0]
    bblock00b = model.blocks[1][0].convs[1]
    bblock00c = model.blocks[1][0].convs[2]
    bblock00s = model.blocks[1][0].shortcut
    # 2. The intermediate results.
    b00a_true = backbone_debug_outs['block00a']
    b00b_true = backbone_debug_outs['block00b']
    b00c_true = backbone_debug_outs['block00c']
    b00s_true = backbone_debug_outs['block00s']
    # 3. The final result.
    b00_true = backbone_debug_outs['block00']
    with torch.no_grad():
        # 1. Step-by-step.
        b00a = bblock00a(conv1_x)
        b00b = bblock00b(b00a)
        b00c = bblock00c(b00b)
        b00s = bblock00s(conv1_x)
        assert torch.dist(b00a, b00a_true) < 1e-12
        assert torch.dist(b00b, b00b_true) < 1e-12
        assert torch.dist(b00c, b00c_true) < 1e-12
        assert torch.dist(b00s, b00s_true) < 1e-12
        # 2. The final result.
        b00 = bblock00(conv1_x)
        assert torch.dist(b00, b00_true) < 1e-12

    # Check the implementation of the bottleneck block (0, 1).
    #
    # 1. Get the blocks.
    bblock01 = model.blocks[1][1]
    bblock01a = model.blocks[1][1].convs[0]
    bblock01b = model.blocks[1][1].convs[1]
    bblock01c = model.blocks[1][1].convs[2]
    # 2. The intermediate results.
    b01a_true = backbone_debug_outs['block01a']
    b01b_true = backbone_debug_outs['block01b']
    b01c_true = backbone_debug_outs['block01c']
    # 3. The final result.
    b01_true = backbone_debug_outs['block01']
    with torch.no_grad():
        # 1. Step-by-step.
        b01a = bblock01a(b00)
        b01b = bblock01b(b01a)
        b01c = bblock01c(b01b)
        assert torch.dist(b01a, b01a_true) < 1e-12
        assert torch.dist(b01b, b01b_true) < 1e-12
        assert torch.dist(b01c, b01c_true) < 1e-12
        # 2. The final result.
        b01 = bblock01(b00)
        assert torch.dist(b01, b01_true) < 1e-12


    # Check the implementation of the bottleneck block (0, 2).
    #
    # 1. Get the blocks.
    bblock02 = model.blocks[1][2]
    bblock02a = model.blocks[1][2].convs[0]
    bblock02b = model.blocks[1][2].convs[1]
    bblock02c = model.blocks[1][2].convs[2]
    # 2. The intermediate results.
    b02a_true = backbone_debug_outs['block02a']
    b02b_true = backbone_debug_outs['block02b']
    b02c_true = backbone_debug_outs['block02c']
    # 3. The final result.
    b02_true = backbone_debug_outs['block02']
    with torch.no_grad():
        # 1. Step-by-step.
        b02a = bblock02a(b01)
        b02b = bblock02b(b02a)
        b02c = bblock02c(b02b)
        assert torch.dist(b02a, b02a_true) < 1e-12
        assert torch.dist(b02b, b02b_true) < 1e-12
        assert torch.dist(b02c, b02c_true) < 1e-12
        # 2. The final result.
        b02 = bblock02(b01)
        assert torch.dist(b02, b02_true) < 1e-12

    # Check t
    res = [model.blocks[i] for i in range(1, len(model.blocks))]
    res_true = intermediate_outs['res_layers']
    with torch.no_grad():
        res_true = intermediate_outs['res_layers']

        res0 = model.blocks[1](conv1_x)
        res1 = model.blocks[2](res0)
        res2 = model.blocks[3](res1)
        res3 = model.blocks[4](res2)

        res = [res0, res1, res2, res3]
        for i, (res_i, res_i_true) in enumerate(zip(res, res_true)):
            diff = torch.dist(res_i, res_i_true)
            logger.info(f"[{i}] diff = {diff}")
            assert diff < 1e-12

    backbone_true = intermediate_outs['backbone']['out']
    with torch.no_grad():
        for out, out_true in zip(res[1:], backbone_true):
            assert torch.dist(out, out_true) < 1e-12


def test_rtdetrv2_resnet_backbone_from_config():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH)
    intermediate_outs = data['intermediate']

    backbone = RTDETRConfig.backbone.make_model()
    ckpt.load_backbone(backbone)
    backbone = freeze_batch_norm(backbone)

    x = data['input']

    backbone_outs = backbone(x)
    backbone_outs_true = intermediate_outs['backbone']['out']
    with torch.no_grad():
        for out, out_true in zip(backbone_outs[-3:], backbone_outs_true):
            assert torch.dist(out, out_true) < 1e-12
