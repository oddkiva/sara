import torch

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch import DEFAULT_DEVICE
from oddkiva.brahma.torch.utils.freeze import freeze_batch_norm
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.checkpoint import RTDETRV2Checkpoint
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.config import RTDETRConfig
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.model import RTDETRv2


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')
DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.data.pt')


def relative_error(a: torch.Tensor, b: torch.Tensor):
    num = torch.dist(a, b)
    denom = torch.norm(a)
    return num / denom


def test_model_from_config_detailed():
    # THE DATA
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))
    intermediate_outs = data['intermediate']

    # THE MODEL
    config = RTDETRConfig()
    model = RTDETRv2(config)

    # LOAD THE MODEL
    ckpt.load_model(model)
    model = freeze_batch_norm(model)

    # THE INPUT
    x = data['input']

    # CHECK THE COMPUTATIONS.
    backbone_outs = model.backbone(x)
    backbone_outs_true = intermediate_outs['backbone']['out']
    with torch.no_grad():
        for out, out_true in zip(backbone_outs[-3:], backbone_outs_true):
            assert torch.dist(out, out_true) < 1e-12

    encoding_pyramid = model.encoder(backbone_outs[-3:])
    encoding_pyramid_true = intermediate_outs['encoder']['out']
    for out, out_true in zip(encoding_pyramid, encoding_pyramid_true):
        assert torch.dist(out, out_true) < 2e-3
        assert torch.dist(out, out_true, p=torch.inf) < 2e-4

    (top_queries,
     _,
     top_geometry_logits,
     memory) = model.query_selector(encoding_pyramid)
    (top_queries_true, top_geometry_logits_true,
     _, _) = intermediate_outs['decoder']['_get_decoder_input']

    assert torch.dist(top_queries, top_queries_true) < 1e-4
    assert torch.dist(top_geometry_logits, top_geometry_logits_true) < 2.5e-3

    value = memory
    value_mask = None
    value_pyramid_hw_sizes = [
        encoding_map.shape[2:]
        for encoding_map in encoding_pyramid
    ]
    (detection_boxes, detection_class_logits,
     dn_boxes, dn_class_logits,
     dn_groups) = model.decoder(
         top_queries.detach(), top_geometry_logits.detach(),
         value, value_pyramid_hw_sizes,
         value_mask=value_mask
     )

    layers_gt = intermediate_outs['decoder']['decoder.layer-by-layer']
    detection_boxes_true = torch.stack(layers_gt['dec_out_bboxes'])
    detection_class_logits_true = torch.stack(layers_gt['dec_out_logits'])

    assert relative_error(detection_boxes, detection_boxes_true) < 6e-4
    assert relative_error(detection_class_logits, detection_class_logits_true) < 6e-4
    assert dn_boxes is None
    assert dn_class_logits is None

    # Check the input denoising groups.
    assert dn_groups.labels is None
    assert dn_groups.geometries is None
    assert dn_groups.attention_mask is None
    assert dn_groups.positive_indices is None
    assert dn_groups.group_count is None
    assert dn_groups.partition is None


def test_model_from_config():
    # THE DATA
    device = torch.device(DEFAULT_DEVICE)
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, device)
    data = torch.load(DATA_FILEPATH, device)
    intermediate_outs = data['intermediate']

    # THE MODEL
    config = RTDETRConfig()
    model = RTDETRv2(config).to(device)

    # LOAD THE MODEL
    ckpt.load_model(model)
    model = freeze_batch_norm(model)

    # THE INPUT
    x = data['input']

    # Check the computations
    box_geometries, box_class_logits, _ = model(x)

    layers_gt = intermediate_outs['decoder']['decoder.layer-by-layer']
    box_geometries_true = torch.stack(layers_gt['dec_out_bboxes'])
    box_class_logits_true = torch.stack(layers_gt['dec_out_logits'])

    assert relative_error(box_geometries, box_geometries_true) < 6e-4
    assert relative_error(box_class_logits, box_class_logits_true) < 6e-4
