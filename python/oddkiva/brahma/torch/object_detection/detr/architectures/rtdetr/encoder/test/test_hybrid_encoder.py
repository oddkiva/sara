import torch

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.checkpoint import (
        RTDETRV2Checkpoint,
        TopDownFusionNet,
        BottomUpFusionNet
    )

from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.encoder import HybridEncoder


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')
DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.data.pt')


def test_hybrid_encoder_construction():
    input_feature_dims = [512, 1024, 2048]
    attn_head_count = 8
    hidden_dim = 256
    attn_feedforward_dim = 1024
    attn_dropout = 0.
    attn_num_layers = 1

    encoder = HybridEncoder(input_feature_dims,
                            attn_head_count,
                            hidden_dim,
                            attn_feedforward_dim=attn_feedforward_dim,
                            attn_dropout=attn_dropout,
                            attn_num_layers=attn_num_layers)
    assert len(encoder.aifi.transformer_encoder.layers) == 1


    stack_count =  len(input_feature_dims) - 1
    assert type(encoder.ccff.fuse_topdown) is TopDownFusionNet
    assert len(encoder.ccff.fuse_topdown.lateral_convs) == stack_count
    assert len(encoder.ccff.fuse_topdown.fusion_blocks) == stack_count

    assert type(encoder.ccff.refine_bottomup) is BottomUpFusionNet
    assert len(encoder.ccff.refine_bottomup.downsample_convs) == stack_count
    assert len(encoder.ccff.refine_bottomup.fusion_blocks) == stack_count


def test_hybrid_encoder_computations():
    input_feature_dims = [512, 1024, 2048]
    attn_head_count = 8
    hidden_dim = 256
    attn_feedforward_dim = 1024
    attn_dropout = 0.
    attn_num_layers = 1

    encoder = HybridEncoder(input_feature_dims,
                            attn_head_count,
                            hidden_dim,
                            attn_feedforward_dim=attn_feedforward_dim,
                            attn_dropout=attn_dropout,
                            attn_num_layers=attn_num_layers)
    assert len(encoder.aifi.transformer_encoder.layers) == 1

    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    encoder = ckpt.load_encoder()
