from oddkiva.brahma.torch.utils.freeze import freeze_batch_norm
import torch

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.checkpoint import RTDETRV2Checkpoint
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.encoder.ccff import (
        BottomUpFusionNet,
        DownsampleConvolution,
        FusionBlock,
    )


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')
DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.data.pt')


def test_downsample_convolution_construction():
    in_channels_sequence = [512, 1024, 2048]
    hidden_dim = 256
    for in_channels in in_channels_sequence:
        block = DownsampleConvolution(in_channels, hidden_dim)

        assert len(block.layers) == 3
        assert type(block.layers[0]) is torch.nn.Conv2d
        assert block.layers[0].in_channels == in_channels
        assert block.layers[0].out_channels == hidden_dim
        assert block.layers[0].kernel_size == (3, 3)
        assert block.layers[0].stride == (2, 2)

        assert type(block.layers[1]) is torch.nn.BatchNorm2d
        assert block.layers[1].num_features == hidden_dim

        assert type(block.layers[2]) is torch.nn.SiLU

def test_downsample_convolution_computations():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    ins = data['intermediate']['encoder']['ccff']['top_down']['fpn']
    bu_outs_true = data['intermediate']['encoder']['ccff']['bottom_up']
    ds_outs_true = bu_outs_true['downsample_convs']
    assert len(ds_outs_true) == 2

    # Load the blocks
    ds_convs = ckpt.load_encoder_downsample_convs()
    ds_convs = freeze_batch_norm(ds_convs)
    assert len(ds_convs) == 2

    # Just check the first one for now
    out = ds_convs[0](ins[0])
    out_true = ds_outs_true[0]
    assert torch.dist(out, out_true) < 1e-12

def test_bottom_up_fusion_blocks():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    ins = data['intermediate']['encoder']['ccff']['top_down']['fpn']
    bu_outs_true = data['intermediate']['encoder']['ccff']['bottom_up']
    ds_outs_true = bu_outs_true['downsample_convs']
    pan_outs_true = bu_outs_true['pan']

    # Load the blocks
    ds_convs = ckpt.load_encoder_downsample_convs()
    fusion = ckpt.load_encoder_bottom_up_fusion_blocks()

    # Freeze the batch norm layers.
    ds_convs = freeze_batch_norm(ds_convs)
    fusion = freeze_batch_norm(fusion)

    assert len(ds_convs) == 2
    assert len(fusion) == 2
    num_steps = len(fusion)

    F_ds = []
    F_topdown_enriched = ins
    F_bottomup_refined = [ins[0]]
    for step in range(num_steps):
        # Take the last feature map.
        F_fine = F_bottomup_refined[step]
        F_coarse = F_topdown_enriched[step + 1]

        # Downsample the fine enriched map.
        F_fine_downsampled = ds_convs[step](F_fine)
        F_ds.append(F_fine_downsampled)

        F_coarse_refined = fusion[step](F_fine_downsampled, F_coarse)
        F_bottomup_refined.append(F_coarse_refined)

    for out, out_true in zip(F_ds, ds_outs_true):
        assert torch.dist(out, out_true) < 1e-12
    for out, out_true in zip(F_bottomup_refined, pan_outs_true):
        assert torch.dist(out, out_true) < 1e-12

def test_bottom_up_fusion_network():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    ins = data['intermediate']['encoder']['ccff']['top_down']['fpn']
    bu_outs_true = data['intermediate']['encoder']['ccff']['bottom_up']
    pan_outs_true = bu_outs_true['pan']

    bu_fusion = BottomUpFusionNet(
        512, 256, 2,
        hidden_dim_expansion_factor=1.0,
        repvgg_stack_depth=3,
        activation='silu'
    )
    ckpt.load_encoder_bottom_up_fusion_network(bu_fusion)
    bu_fusion = freeze_batch_norm(bu_fusion)

    assert len(bu_fusion.downsample_convs) == 2
    assert len(bu_fusion.fusion_blocks) == 2

    pan_outs = bu_fusion(ins)

    for out, out_true in zip(pan_outs, pan_outs_true):
        assert torch.dist(out, out_true) < 1e-12
