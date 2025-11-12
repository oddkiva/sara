import torch

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.utils.freeze import freeze_batch_norm
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.encoder.backbone_feature_pyramid_projection import (
        BackboneFeaturePyramidProjection
    )
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.checkpoint import (
        UnbiasedConvBNA,
        RTDETRV2Checkpoint
    )


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')
DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.data.pt')


class FeaturePyramidProjection:
    hidden_dim: int = 256


def test_backbone_feature_pyramid_projection_construction():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))

    backbone = ckpt.load_backbone()

    fp_dims = backbone.feature_pyramid_dims[-3:]
    fp_proj = BackboneFeaturePyramidProjection(
        fp_dims,
        FeaturePyramidProjection.hidden_dim
    )
    fp_proj = freeze_batch_norm(fp_proj)


def test_backbone_feature_pyramid_projection_computations():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    fp_proj = ckpt.load_encoder_input_proj()

    # Check the computations with the data.
    backbone_outs = data['intermediate']['backbone']
    fp_proj_outs_true = data['intermediate']['encoder']['input_proj']
    fp_proj_outs = fp_proj(backbone_outs)

    with torch.no_grad():
        for fp_proj_out, fp_proj_out_true in zip(fp_proj_outs, fp_proj_outs_true):
            diff = torch.norm(fp_proj_out - fp_proj_out_true)
            print(diff)
            assert diff < 1e-12
