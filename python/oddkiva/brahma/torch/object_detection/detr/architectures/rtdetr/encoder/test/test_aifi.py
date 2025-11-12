from loguru import logger

import torch

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.checkpoint import RTDETRV2Checkpoint
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.encoder.aifi import AIFI


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')
DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.data.pt')


class AIFIConfig:
    hidden_dim: int = 256
    head_count: int  = 8
    feedforward_dim: int = 1024
    dropout: float = 0.
    use_encoder_idx: list[int] = [2]
    num_encoder_layers: int = 1
    normalize_before: bool = False
    encoder_activation: torch.nn.Module | None = torch.nn.GELU()


def test_aifi():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    aifi = ckpt.load_encoder_aifi()

    input_proj_out = data['intermediate']['encoder']['input_proj']
    f5_out = input_proj_out[-1]
    aifi_out_true = data['intermediate']['encoder']['aifi']

    aifi_out = aifi.forward(f5_out)

    assert aifi_out.shape == aifi_out_true.shape
    diff = torch.norm(aifi_out - aifi_out_true)
    logger.debug(f'diff = {diff}')
    assert diff < 1e-12

    import IPython; IPython.embed()
