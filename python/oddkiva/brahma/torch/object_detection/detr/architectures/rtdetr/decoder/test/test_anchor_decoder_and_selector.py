from oddkiva.brahma.torch.utils.freeze import freeze_batch_norm
import torch

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.checkpoint import RTDETRV2Checkpoint
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.anchor_decoder import AnchorGeometryLogitEnumerator
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.anchor_decoder import AnchorDecoder
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.anchor_selector import AnchorSelector


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')
DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.data.pt')


def test_anchor_logit_enumerator():
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    fpyr_projected: list[torch.Tensor] = \
        data['intermediate']['decoder']['input_proj']
    fpyr_image_sizes = [fmap.shape[2:][::-1]
                        for fmap in fpyr_projected]

    anchor_logit_enumerator = AnchorGeometryLogitEnumerator(
        normalized_base_size=0.05,
        eps=1e-2
    )

    anchor_logits, anchor_mask = anchor_logit_enumerator(
        fpyr_image_sizes,
        fpyr_projected[0].device
    )
    assert anchor_logits.requires_grad is False
    assert anchor_mask.requires_grad is False

    anchor_logits_true, anchor_mask_true = \
        data['intermediate']['decoder']['_generate_anchors']

    # Filter out the invalid rows with this recipe...
    valid_anchor_logits = anchor_logits[anchor_mask[:, 0], :]
    valid_anchor_logits_true = \
        anchor_logits_true[0, anchor_mask_true[0, :, 0], :]

    assert torch.equal(anchor_mask_true[0], anchor_mask)
    assert torch.dist(valid_anchor_logits, valid_anchor_logits_true) < 2.5e-5


def test_anchor_decoder():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    encoding_dim = 256
    hidden_dim = 256
    num_classes = 80

    anchor_decoder = AnchorDecoder(
        encoding_dim, hidden_dim, num_classes,
        geometry_head_layer_count=3,
        geometry_head_activation='relu',
        normalized_base_size=0.05,
        logit_eps=1e-2,
        precalculate_anchor_geometry_logits=True,
        image_pyramid_wh_sizes=[(80, 80), (40, 40), (20, 20)],
        device = torch.device('cpu'),
        initial_class_probability=0.1
    )
    ckpt.load_decoder_anchor_decoder(anchor_decoder)
    assert type(anchor_decoder) is AnchorDecoder

    # -------------------------------------------------------------------------
    # Step 1: check the anchor geometry logits
    # -------------------------------------------------------------------------
    fpyr_projected: list[torch.Tensor] = \
        data['intermediate']['decoder']['input_proj']
    fpyr_image_sizes = [(fmap.shape[3], fmap.shape[2])
                        for fmap in fpyr_projected]
    anchor_logits, anchor_mask = \
        anchor_decoder.anchor_geometry_logit_enumerator(
            fpyr_image_sizes,
            fpyr_projected[0].device
        )
    anchor_geometry_logits_true, anchor_mask_true = \
        data['intermediate']['decoder']['_generate_anchors']

    # Filter out the invalid rows with this recipe...
    valid_anchor_logits = anchor_logits[anchor_mask[:, 0], :]
    valid_anchor_logits_true = \
        anchor_geometry_logits_true[0, anchor_mask_true[0, :, 0], :]

    assert torch.equal(anchor_mask_true[0], anchor_mask)
    assert torch.dist(valid_anchor_logits, valid_anchor_logits_true) < 2.5e-5


    # -------------------------------------------------------------------------
    # Step 2: construct the memory by hand
    # -------------------------------------------------------------------------
    memory = torch.cat([
        fmap.flatten(2).permute(0, 2, 1)
        for fmap in fpyr_projected
    ], dim=1)


    # -------------------------------------------------------------------------
    # Step 3: decode the memory into anchor class logits and anchor geometry
    # logits.
    # -------------------------------------------------------------------------
    _, anchor_class_logits, anchor_geom_logits_refined = \
        anchor_decoder.forward(memory, fpyr_image_sizes)


    anchor_class_logits_true, anchor_geom_logits_refined_true = \
        data['intermediate']['decoder']['_get_decoder_input_part_1']

    valid_geom_logits = anchor_geom_logits_refined[0, anchor_mask[:, 0], :]
    valid_geom_logits_true = \
        anchor_geom_logits_refined_true[0, anchor_mask_true[0, :, 0], :]

    assert torch.dist(anchor_class_logits, anchor_class_logits_true) < 1e-12
    assert torch.dist(valid_geom_logits, valid_geom_logits_true) < 2.5e-5


def test_anchor_selector():
    # THE DATA
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))
    decoder_data = data['intermediate']['decoder']

    # THE MODEL
    encoding_dim = 256
    hidden_dim = 256
    num_classes = 80

    anchor_decoder = AnchorDecoder(
        encoding_dim, hidden_dim, num_classes,
        geometry_head_layer_count=3,
        geometry_head_activation='relu',
        normalized_base_size=0.05,
        logit_eps=1e-2,
        precalculate_anchor_geometry_logits=True,
        image_pyramid_wh_sizes=[(80, 80), (40, 40), (20, 20)],
        device = torch.device('cpu'),
        initial_class_probability=0.1
    )
    anchor_selector = AnchorSelector(top_K=300)

    # Load the model weights.
    ckpt.load_decoder_anchor_decoder(anchor_decoder)

    # Reconstruct the memory
    fpyr_projected: list[torch.Tensor] = decoder_data['input_proj']
    memory = torch.cat([
        fmap.flatten(2).permute(0, 2, 1)
        for fmap in fpyr_projected
    ], dim=1)
    memory_wh_sizes = [
        (fmap.shape[3], fmap.shape[2])
        for fmap in fpyr_projected
    ]

    (memory_filtered,
     anchor_class_logits,
     anchor_geom_logits) = anchor_decoder.forward(memory, memory_wh_sizes)


    # Fetch the class logits and geometry logits for each each anchor
    anchor_class_logits, anchor_geom_logits = \
        decoder_data['_get_decoder_input_part_1']

    (top_queries,
     top_class_logits,
     top_geom_logits) = anchor_selector.forward(memory_filtered,
                                                anchor_class_logits,
                                                anchor_geom_logits)


    (top_queries_true, top_geom_logits_true,
     _, _) = decoder_data['_get_decoder_input']

    assert torch.dist(top_queries, top_queries_true) < 1e-12
    assert torch.dist(top_geom_logits, top_geom_logits_true) < 2.5e-3
