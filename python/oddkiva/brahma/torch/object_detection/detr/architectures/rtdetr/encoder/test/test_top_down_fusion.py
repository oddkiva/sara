import torch
import torch.nn.functional as F

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.checkpoint import (RTDETRV2Checkpoint,
                              TopDownFusionNet)


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')
DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.data.pt')


def test_top_down_fusion_details():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    # Load the blocks
    lateral_convs = ckpt.load_encoder_lateral_convs()
    fpn = ckpt.load_encoder_top_down_fusion_blocks()
    assert len(lateral_convs) == 2
    assert len(fpn) == 2

    # Load the data.
    S = data['intermediate']['encoder']['input_proj']
    F5 = data['intermediate']['encoder']['aifi']['out']

    top_down_outs = data['intermediate']['encoder']['ccff']['top_down']
    fpn_outs_true = top_down_outs['fpn']

    n, _, h, w = S[-1].shape
    _, _, c = F5.shape
    F5_map = F5.permute(0, 2, 1).reshape((n, c, h, w))

    # # Outputs
    F_enriched = [F5_map]

    num_steps = len(fpn)
    assert num_steps == len(S) - 1

    with torch.no_grad():
        for step in range(num_steps):
            print("step = ", step)
            # Take the last feature map.
            #
            # Calculate the lateral convolution, this will be the real enriched
            # feature map.
            F_enriched[-1] = lateral_convs[step](F_enriched[-1])
            S_fine = S[num_steps - 1 - step]

            # Upscale the coarse enriched map.
            F_coarse = F_enriched[-1]
            F_coarse_upscaled = F.interpolate(F_coarse, scale_factor=2,
                                              mode='nearest')

            # Calculate the fine enriched map.
            FS = fpn[step](F_coarse_upscaled, S_fine)
            F_enriched.append(FS)

        F_enriched.reverse()

        for fpn_out, fpn_out_true in zip(F_enriched, fpn_outs_true):
            fpn_diff = torch.dist(fpn_out, fpn_out_true)
            print(f'fpn_diff = {fpn_diff}')
            assert fpn_diff < 1e-12


def test_top_down_fusion_module():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    # Load the blocks
    fpn = TopDownFusionNet(
        512, 256, 2,
        hidden_dim_expansion_factor=1.0,
        repvgg_stack_depth=3,
        activation='silu'
    )
    ckpt.load_encoder_top_down_fusion_network(fpn)

    # Load the data.
    S = data['intermediate']['encoder']['input_proj']
    F5_flat = data['intermediate']['encoder']['aifi']['out']

    top_down_outs = data['intermediate']['encoder']['ccff']['top_down']
    fpn_outs_true = top_down_outs['fpn']

    n, _, h, w = S[-1].shape
    _, _, c = F5_flat.shape
    F5 = F5_flat.permute(0, 2, 1).reshape((n, c, h, w))

    with torch.no_grad():
        fpn_outs = fpn.forward(F5, S)
        assert len(fpn_outs) == 3

        for out, out_true in zip(fpn_outs, fpn_outs_true):
            assert torch.dist(out, out_true) < 1e-12
