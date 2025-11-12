from loguru import logger

import torch

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.checkpoint import RTDETRV2Checkpoint


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
    s5 = input_proj_out[-1]
    s5_flat = s5.flatten(2).permute(0, 2, 1)

    h, w = s5.shape[2:]
    s5_pe = aifi.positional_encoding_fn.forward((w, h))
    s5_pe = s5_pe.flatten(0, 1)[None, ...]
    s5_pe_true = data['intermediate']['encoder']['aifi']['debug']['pos_embed']
    self_attn_out_true = \
        data['intermediate']['encoder']['aifi']['debug']['self_attn']
    norm1_out_true = \
        data['intermediate']['encoder']['aifi']['debug']['norm1']
    ffn_out_true = \
        data['intermediate']['encoder']['aifi']['debug']['ffn']
    norm2_out_true = \
        data['intermediate']['encoder']['aifi']['debug']['norm2']

    with torch.no_grad():
        assert s5_pe.shape == s5_pe_true.shape
        assert torch.max(torch.abs(s5_pe - s5_pe_true)) < 2e-6
        assert torch.norm(s5_pe - s5_pe_true) < 1e-4

        aifi_layer = aifi.transformer_encoder.layers[0]
        self_attn = aifi_layer.self_attention

        # Check the self attention
        Q = K = s5_flat + s5_pe
        V = s5_flat
        self_attn_out, _ = self_attn.forward(Q, K, value=V, attn_mask=None)
        assert torch.norm(self_attn_out - self_attn_out_true) < 5e-5

        # Check the layer norm 1.
        V_residuals = aifi_layer.dropout_1(self_attn_out)
        V_enhanced = aifi_layer.layer_norm_1(V + V_residuals)
        norm1_out = V_enhanced
        assert torch.norm(norm1_out - norm1_out_true) < 8e-5

        # Check the FFN.
        ffn_out = aifi_layer.feedforward(norm1_out)
        assert torch.norm(ffn_out - ffn_out_true) < 8e-5


        # Check the layer norm 2.
        norm2_out = aifi_layer.layer_norm_2(
            norm1_out + aifi_layer.dropout_2(ffn_out)
        )
        assert torch.norm(norm2_out - norm2_out_true.data) < 8e-5

    aifi_out_true = data['intermediate']['encoder']['aifi']['out']
    with torch.no_grad():
        assert torch.norm(norm2_out - aifi_out_true) < 8e-5

        aifi_out = aifi.forward(s5)
        assert aifi_out.shape == aifi_out_true.shape

        max_coeff_dist = torch.max(torch.abs(aifi_out - aifi_out_true))
        dist = torch.norm(aifi_out - aifi_out_true)
        logger.debug(f'max_coeff_dist = {max_coeff_dist}')
        logger.debug(f'dist = {dist}')
        assert dist < 8e-5

