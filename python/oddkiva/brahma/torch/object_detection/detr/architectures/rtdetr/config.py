import torch

from oddkiva.brahma.torch.backbone.resnet.rtdetrv2_variant import (
    ResNet50RTDETRV2Variant
)
from oddkiva.brahma.torch.object_detection.detr.architectures\
    .rtdetr.encoder.hybrid_encoder import HybridEncoder
from oddkiva.brahma.torch.object_detection.detr.architectures\
    .rtdetr.decoder.query_selector import QuerySelector
from oddkiva.brahma.torch.object_detection.detr.architectures\
    .rtdetr.decoder.query_decoder import MultiScaleDeformableTransformerDecoder


class BackboneConfig:
    Model = ResNet50RTDETRV2Variant

    @staticmethod
    def make_model() -> Model:
        model = BackboneConfig.Model()
        return model


class EncoderConfig:
    """
    This class encapsulates the default parameters for RT-DETR v2's hybrid
    encoder which basically sequences three modules:
    (`FeaturePyramidProjection` -> `AIFI` -> `CCFF`).
    """
    Model = HybridEncoder

    """
    AIFI parameters.
    """
    # These parameters are the feature pyramid dimensions of RT-DETR's
    # ResNet-50 variant.
    input_feature_dims = [512, 1024, 2048]
    # The feature maps are projected onto the same vector space.
    encoding_dim = 256
    # The parameters of AIFI's self-attention module.
    attn_head_count = 8
    attn_feedforward_dim = 1024
    attn_dropout = 0.
    attn_num_layers = 1

    """
    CCFF parameters.

    TODO
    """

    # Asserts
    assert encoding_dim % attn_head_count == 0

    @staticmethod
    def make_model() -> Model:
        C = EncoderConfig
        return EncoderConfig.Model(
            C.input_feature_dims,
            C.attn_head_count,
            C.encoding_dim,
            C.attn_feedforward_dim,
            C.attn_dropout,
            C.attn_num_layers
        )


class QuerySelectorConfig:
    Model = QuerySelector

    # The initial anchor geometries.
    anchor_normalized_base_box_sizes = 0.05
    anchor_geometry_logit_eps = 0.01
    precalculate_anchor_geometry_logits = True

    # The query input dimension is the AIFI's output query dimension
    encoding_dim = 256
    hidden_dim = 256
    # 80 -> COCO dataset
    query_num_classes = 80
    # For inference.
    query_pyramid_wh_sizes = [(80, 80), (40, 40), (20, 20)]

    geometry_head_layer_count = 3
    geometry_head_activation = 'relu'
    initial_object_class_probability = 0.1

    top_K = 300

    @staticmethod
    def make_model() -> Model:
        C = QuerySelectorConfig
        return QuerySelectorConfig.Model(
            C.encoding_dim,
            C.hidden_dim,
            C.query_num_classes,
            C.query_pyramid_wh_sizes,
            C.top_K,
            geometry_head_layer_count=C.geometry_head_layer_count,
            geometry_head_activation=C.geometry_head_activation,
            initial_object_class_probability=C.initial_object_class_probability,
            precalculate_anchor_geometry_logits=C.precalculate_anchor_geometry_logits,
            anchor_normalized_base_size=C.anchor_normalized_base_box_sizes,
            anchor_logit_eps=C.anchor_geometry_logit_eps
        )


class DecoderConfig:
    Model = MultiScaleDeformableTransformerDecoder

    hidden_dim = 256
    kv_count_per_level = [4, 4, 4]
    num_classes = 80
    attn_value_dim = 32
    attn_head_count = 8
    attn_num_layers = 6
    attn_dropout = 0.0
    attn_feedforward_dim = 1024
    normalize_before = False

    @staticmethod
    def make_model() -> Model:
        C = DecoderConfig
        return DecoderConfig.Model(
            C.hidden_dim,
            C.attn_value_dim,
            C.kv_count_per_level,
            num_classes=C.num_classes,
            attn_head_count=C.attn_head_count,
            attn_num_layers=C.attn_num_layers,
            attn_dropout=C.attn_dropout,
            attn_feedforward_dim=C.attn_feedforward_dim,
            normalize_before=C.normalize_before
        )


class RTDETRConfig:
    backbone = BackboneConfig()
    encoder = EncoderConfig()
    query_selector = QuerySelectorConfig()
    decoder = DecoderConfig()

    assert encoder.encoding_dim == query_selector.encoding_dim
    assert query_selector.hidden_dim == decoder.hidden_dim
