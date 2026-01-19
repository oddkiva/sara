import torch

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.utils.freeze import freeze_batch_norm
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.config import RTDETRConfig
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.encoder.feature_pyramid_projection import (
        FeaturePyramidProjection
    )
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.checkpoint import (
        RTDETRV2Checkpoint
    )


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')
DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.data.pt')


def test_backbone_feature_pyramid_projection_computations():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    fp_proj = FeaturePyramidProjection(
        [512, 1024, 2048],
        256
    )

    ckpt.load_encoder_input_proj(fp_proj)
    fp_proj = freeze_batch_norm(fp_proj)

    # Check the computations with the data.
    backbone_outs = data['intermediate']['backbone']['out']
    fp_proj_outs_true = data['intermediate']['encoder']['input_proj']
    fp_proj_outs = fp_proj(backbone_outs)

    with torch.no_grad():
        for fp_proj_out, fp_proj_out_true in zip(fp_proj_outs, fp_proj_outs_true):
            diff = torch.dist(fp_proj_out, fp_proj_out_true)
            assert diff < 1e-12
