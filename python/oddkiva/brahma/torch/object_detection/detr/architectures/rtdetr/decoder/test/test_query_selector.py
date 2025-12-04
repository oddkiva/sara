from oddkiva.brahma.torch.utils.freeze import freeze_batch_norm
import torch

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.checkpoint import RTDETRV2Checkpoint
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.query_selector import QuerySelector
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.config import RTDETRConfig


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')
DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.data.pt')


def test_query_selector():
    # THE DATA
    device = torch.device('mps:0')
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, map_location=device)
    data = torch.load(DATA_FILEPATH, device)
    decoder_data = data['intermediate']['decoder']

    # THE MODEL
    query_selector = RTDETRConfig.query_selector.make_model()
    query_selector = query_selector.to(device)
    ckpt.load_query_selector(query_selector)
    query_selector = freeze_batch_norm(query_selector)
    assert type(query_selector) is QuerySelector

    encoder_outs = data['intermediate']['encoder']['out']
    for encoder_out in encoder_outs:
        assert encoder_out.requires_grad is True

    (top_queries,
     top_class_logits,
     top_geometry_logits,
     _) = query_selector.forward(encoder_outs)

    assert top_queries.requires_grad is True
    assert top_class_logits.requires_grad is True
    assert top_geometry_logits.requires_grad is True

    (top_queries_true, top_geometry_logits_true,
     _, _) = decoder_data['_get_decoder_input']

    assert torch.dist(top_queries, top_queries_true) < 1e-4
    assert torch.dist(top_geometry_logits, top_geometry_logits_true) < 2.5e-3
