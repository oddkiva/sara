import torch

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.checkpoint import RTDETRV2Checkpoint
from oddkiva.brahma.torch.object_detection.common.anchors import (
    calculate_anchor_logits,
    enumerate_pyramid_anchors
)
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder import MultiScaleDeformableTransformerDecoder


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')
DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.data.pt')


def test_decoder_construction():
    encoding_dim = 256
    hidden_dim = 256
    kv_count_per_level = [4, 4, 4]
    attn_head_count = 8
    attn_feedforward_dim = 1024
    attn_dropout = 0.0
    attn_num_layers = 6
    activation = 'relu'
    normalize_before = False
    # Multi-scale deformable attention parameters

    noised_true_boxes_count = 100
    label_noise_ratio = 0.5
    box_noise_scale = 1.0

    decoder = MultiScaleDeformableTransformerDecoder(
        encoding_dim,
        hidden_dim,
        kv_count_per_level,
        attn_head_count=attn_head_count,
        attn_feedforward_dim=attn_feedforward_dim,
        attn_num_layers=attn_num_layers,
        attn_dropout=attn_dropout,
        normalize_before=normalize_before
    )
    assert len(decoder.feature_projectors) == 3


def test_decoder_anchor_generation():
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))
    anchor_logits_true, anchor_valid_mask_true = \
        data['intermediate']['decoder']['_generate_anchors']

    pyramid_image_sizes = [(80, 80), (40, 40), (20, 20)]
    normalized_base_box_size = 0.05  # percentage in w.r.t. image sizes
    device = torch.device('cpu')

    anchor_pyramid = enumerate_pyramid_anchors(
        pyramid_image_sizes,
        normalized_base_box_size=normalized_base_box_size,
        normalize_anchor_geometry=True,
        device=device
    )
    anchors = torch.cat(anchor_pyramid, dim=0)
    anchor_logits, anchor_valid_mask = calculate_anchor_logits(
        anchors, eps=1e-2
    )

    assert torch.equal(anchor_valid_mask_true[0], anchor_valid_mask)

    # Filter out the invalid rows.
    # with this recipe...
    valid_anchor_logits = anchor_logits[anchor_valid_mask[:, 0], :]
    valid_anchor_logits_true = \
        anchor_logits_true[0, anchor_valid_mask_true[0, :, 0], :]
    assert torch.norm(valid_anchor_logits - valid_anchor_logits_true) < 1e-5


def test_decoder_computations():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    decoder = ckpt.load_decoder()

    encoder_out = data['intermediate']['encoder']['out']
    dec_input_proj_outs_true = data['intermediate']['decoder']['input_proj']

    dec_input_proj_outs = decoder.feature_projectors(encoder_out)
    with torch.no_grad():
        for out, out_true in zip(dec_input_proj_outs, dec_input_proj_outs_true):
            assert torch.linalg.vector_norm(out - out_true) < 1e-12
            assert torch.linalg.vector_norm(out - out_true, ord=torch.inf) < 1e-12

    query_pyramid_true, pyramid_image_sizes_true = \
        data['intermediate']['decoder']['_get_encoder_input']
    query_pyramid = [
        fmap.flatten(2).permute(0, 2, 1)
        for fmap in dec_input_proj_outs
    ]
    query_pyramid = torch.cat(query_pyramid, dim=1)

    pyramid_image_sizes = [
        fmap.shape[2:][::-1]
        for fmap in dec_input_proj_outs
    ]
    with torch.no_grad():
        for out, out_true in zip(query_pyramid, query_pyramid_true):
            assert torch.linalg.vector_norm(out - out_true) < 1e-12
    for shape, shape_true in zip(pyramid_image_sizes,
                                 pyramid_image_sizes_true):
        assert shape == torch.Size(shape_true)

