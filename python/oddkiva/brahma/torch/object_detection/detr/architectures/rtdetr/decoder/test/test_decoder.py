import torch

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.checkpoint import RTDETRV2Checkpoint
from oddkiva.brahma.torch.object_detection.common.anchors import enumerate_anchors
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

    queries_true, spatial_shapes_true = \
        data['intermediate']['decoder']['_get_encoder_input']
    queries = [
        fmap.flatten(2).permute(0, 2, 1)
        for fmap in dec_input_proj_outs
    ]
    queries = torch.cat(queries, dim=1)
    wh_sizes = [
        fmap.shape[2:][::-1]
        for fmap in dec_input_proj_outs
    ]
    with torch.no_grad():
        for out, out_true in zip(queries, queries_true):
            assert torch.linalg.vector_norm(out - out_true) < 1e-12
    for shape, shape_true in zip(wh_sizes, spatial_shapes_true):
        assert shape == torch.Size(shape_true)

    anchors, valid_mask = \
        data['intermediate']['decoder']['_generate_anchors']

    # (640, 640)
    # -> (80, 80) = (640 //  8, 640 //  8)
    # -> (40, 40) = (640 // 16, 640 // 16)
    # -> (20, 20) = (640 // 32, 640 // 32)
    #      ^   ^             ^
    #      |---|--size       |----------|---- stride
    #    box

    device = torch.device('cpu')

    relative_box_sizes = [0.05 * (2 ** lvl) for lvl in range(len(wh_sizes))]
    box_sizes = [(f * w, f * h)
                 for f, (w, h) in zip(relative_box_sizes, wh_sizes)]
    anchors = [
        enumerate_anchors(wh, bsizes, True, device)
        for wh, bsizes in zip(wh_sizes, box_sizes)
    ]
    anchors = torch.cat(anchors, dim=1).to(device)
    # For the logits
    eps = 1e-2
    valid_mask = ((anchors > eps) * (anchors < (1 - eps)))\
        .all(-1, keepdim=True)

    # anchor encodings are the logits
    anchor_logits = torch.log(anchors / (1 - anchors))
    anchor_logits = torch.where(valid_mask, anchor_logits, torch.inf)
