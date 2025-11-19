import torch

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.checkpoint import RTDETRV2Checkpoint
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder import MultiScaleDeformableTransformerDecoder
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.anchor_decoder import AnchorGeometryLogitEnumerator
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.anchor_selector import AnchorSelector


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')
DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.data.pt')


def test_decoder_construction():
    hidden_dim = 256
    kv_count_per_level = [4, 4, 4]
    attn_head_count = 8
    attn_feedforward_dim = 1024
    attn_dropout = 0.0
    attn_num_layers = 6
    # activation = 'relu'
    normalize_before = False
    # Multi-scale deformable attention parameters

    # DN-DETR parameters
    # noised_true_boxes_count = 100
    # label_noise_ratio = 0.5
    # box_noise_scale = 1.0

    MultiScaleDeformableTransformerDecoder(
        hidden_dim,
        kv_count_per_level,
        attn_head_count=attn_head_count,
        attn_feedforward_dim=attn_feedforward_dim,
        attn_num_layers=attn_num_layers,
        attn_dropout=attn_dropout,
        normalize_before=normalize_before
    )


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

    anchor_logits_true, anchor_mask_true = \
        data['intermediate']['decoder']['_generate_anchors']

    # Filter out the invalid rows with this recipe...
    valid_anchor_logits = anchor_logits[anchor_mask[:, 0], :]
    valid_anchor_logits_true = \
        anchor_logits_true[0, anchor_mask_true[0, :, 0], :]

    assert torch.equal(anchor_mask_true[0], anchor_mask)
    assert torch.norm(valid_anchor_logits - valid_anchor_logits_true) < 2.5e-5


def test_anchor_decoder():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))
    anchor_decoder = ckpt.load_decoder_anchor_decoder()

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
    assert torch.norm(valid_anchor_logits - valid_anchor_logits_true) < 2.5e-5


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
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    # Reconstruct the memory
    fpyr_projected: list[torch.Tensor] = \
        data['intermediate']['decoder']['input_proj']
    memory = torch.cat([
        fmap.flatten(2).permute(0, 2, 1)
        for fmap in fpyr_projected
    ], dim=1)

    # Fetch the class logits and geometry logits for each each anchor
    anchor_class_logits, anchor_geom_logits = \
        data['intermediate']['decoder']['_get_decoder_input_part_1']

    anchor_selector = AnchorSelector(top_k=300)
    (top_queries,
     top_class_logits,
     top_geom_logits) = anchor_selector.forward(memory,
                                                anchor_class_logits,
                                                anchor_geom_logits)
