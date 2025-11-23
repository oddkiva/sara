import torch

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.checkpoint import RTDETRV2Checkpoint


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')
DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.data.pt')


def test_transformer_decoder():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    memory, _ = data['intermediate']['decoder']['_get_encoder_input']
    init_ref_contents, init_ref_points_unact, _, _ = \
        data['intermediate']['decoder']['_get_decoder_input']
    
    box_geometries_true, box_class_logits_true = \
        data['intermediate']['decoder']['decoder']

    decoder = ckpt.load_transformer_decoder()
    box_geometries, box_class_logits = decoder.forward(
        init_ref_contents, init_ref_points_unact, memory,
        value_mask=None
    )

    assert torch.norm(box_geometries - box_geometries_true) < 1e-12
    assert torch.norm(box_class_logits - box_class_logits_true) < 1e-12
