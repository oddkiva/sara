import torch
import torch.nn as nn

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.encoder.ccff import LateralConvolution
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.checkpoint import RTDETRV2Checkpoint
from torch.nn.modules import SiLU


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')
DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.data.pt')


def test_lateral_convolutions():
    lateral_conv = LateralConvolution(256, 256)

    # Check the construction of the lateral convolutions.
    layers = lateral_conv.layers
    assert type(layers[0]) is nn.Conv2d
    assert layers[0].bias is None
    assert type(layers[1]) is nn.BatchNorm2d
    assert type(layers[2]) is nn.SiLU


def test_lateral_convolution_computations():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    # Load the blocks
    lateral_convs = ckpt.load_encoder_lateral_convs()

    # Load the data.
    S = data['intermediate']['encoder']['input_proj']
    F5 = data['intermediate']['encoder']['aifi']['out']
    lateral_conv_outs_true = data['intermediate']['encoder']['ccff']['top_down']['lateral_convs']

    n, _, h, w = S[-1].shape
    _, _, c = F5.shape
    F5_map = F5.permute(0, 2, 1).reshape((n, c, h, w))

    # Outputs
    F_enriched = [F5_map]
    F_yellowed = []


    # Check the first lateral convs for now. We need to check the FPN blocks as
    # well.
    lateral_conv = lateral_convs[-1]
    lateral_conv_out = lateral_conv(F5_map)
    lateral_conv_out_true = lateral_conv_outs_true[0]
    diff = torch.norm(lateral_conv_out - lateral_conv_out_true)
    print(f'diff = {diff}')
    assert diff < 1e-12
