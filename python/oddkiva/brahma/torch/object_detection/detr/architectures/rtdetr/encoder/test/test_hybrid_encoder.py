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
ENCODER_DEBUG_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                          'encoder.debug.pt')
ENCODER_FROZEN_STATE_DEBUG_FILEPATH = (
    DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' / 'encoder-frozen-state.debug.pt')


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
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    encoder = ckpt.load_encoder()
    assert len(encoder.aifi.transformer_encoder.layers) == 1

    backbone_outs = data['intermediate']['backbone']['out']
    fp_proj_outs_true = data['intermediate']['encoder']['input_proj']
    aifi_out_true = data['intermediate']['encoder']['aifi']['out']
    ccff_out_true = data['intermediate']['encoder']['ccff']['bottom_up']['pan']

    # Project the feature vectors of the feature pyramid into the same
    # dimensional space.
    fp_proj_outs = encoder.backbone_feature_proj(backbone_outs)
    with torch.no_grad():
        for Si, Si_true in zip(fp_proj_outs, fp_proj_outs_true):
            assert torch.norm(Si - Si_true) < 1e-12

    # Perform self-attention of the coarsest feature map of the feature
    # pyramid.
    # [F3, F4, F5] for ResNet
    S = fp_proj_outs
    F5 = encoder.aifi.forward(S[-1])
    with torch.no_grad():
        assert torch.norm(F5 - aifi_out_true) < 1.2e-4

    # The top-down then bottom-up fusion scheme.
    Q = encoder.ccff.forward(F5, S)
    with torch.no_grad():
        for out, out_true in zip(Q, ccff_out_true):
            assert torch.linalg.vector_norm(out - out_true) < 2e-3
            assert torch.linalg.vector_norm(out - out_true, ord=torch.inf) < 5e-5

    # THE WHOLE IMPLEMENTATION
    Q2 = encoder(backbone_outs)
    with torch.no_grad():
        for out, out_true in zip(Q2, ccff_out_true):
            assert torch.linalg.vector_norm(out - out_true) < 2e-3
            assert torch.linalg.vector_norm(out - out_true, ord=torch.inf) < 2e-4

        encoder_out_true = data['intermediate']['encoder']['out']
        for out, out_true in zip(Q2, encoder_out_true):
            assert torch.linalg.vector_norm(out - out_true) < 2e-3
            assert torch.linalg.vector_norm(out - out_true, ord=torch.inf) < 2e-4
