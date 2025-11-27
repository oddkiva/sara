import torch
import torch.nn.functional as F

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.checkpoint import RTDETRV2Checkpoint
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.query_decoder import (
        MultiScaleDeformableTransformerDecoderLayer
    )


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')
DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.data.pt')


def test_transformer_decoder_layer_0():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    decoder_gt = data['intermediate']['decoder']
    layer0_gt = decoder_gt['decoder_layer_0']

    memory, memory_spatial_hw_sizes = decoder_gt['_get_encoder_input']
    query, query_geometry_logits, _, _ = decoder_gt['_get_decoder_input']

    decoder = ckpt.load_transformer_decoder()

    layer = decoder.layers[0]
    assert type(layer) is MultiScaleDeformableTransformerDecoderLayer

    query_geometries = F.sigmoid(query_geometry_logits)
    query_geometries_true = layer0_gt['ref_points_detach']
    assert torch.dist(query_geometries, query_geometries_true) < 1e-12

    qgeom_embeds = \
        decoder.box_geometry_embedding_map(query_geometries)
    qgeom_embeds_true = layer0_gt['query_pos_embed']
    assert torch.dist(qgeom_embeds, qgeom_embeds_true) < 1e-12


    # box_geometries, box_class_logits = layer.forward(
    #     query.detach(), query_geometries,
    #     memory, memory_spatial_hw_sizes,
    #     query_positional_embeds=query_geometry_embeds
    # )


# def test_transformer_decoder():
#     ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
#     data = torch.load(DATA_FILEPATH, torch.device('cpu'))
#
#     memory, memory_spatial_hw_sizes = \
#         data['intermediate']['decoder']['_get_encoder_input']
#     init_ref_contents, init_ref_points_unact, _, _ = \
#         data['intermediate']['decoder']['_get_decoder_input']
#
#     box_geometries_true, box_class_logits_true = \
#         data['intermediate']['decoder']['decoder']
#
#     decoder = ckpt.load_transformer_decoder()
#     box_geometries, box_class_logits = decoder.forward(
#         init_ref_contents.detach(), init_ref_points_unact,
#         memory, memory_spatial_hw_sizes,
#         value_mask=None
#     )
#
#     assert torch.norm(box_geometries - box_geometries_true) < 1e-12
#     assert torch.norm(box_class_logits - box_class_logits_true) < 1e-12
