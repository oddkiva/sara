import torch
import torch.nn as nn

from oddkiva.brahma.torch.backbone.resnet.rtdetrv2_variant import UnbiasedConvBNA
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.anchor_decoder import AnchorDecoder
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.anchor_selector import AnchorSelector
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.query_decoder import MultiScaleDeformableTransformerDecoder
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.encoder.hybrid_encoder import FeaturePyramidProjection


class RTDETRv2Decoder(nn.Module):

    def __init__(self,
                 num_classes: int,
                 encoding_dim: int,
                 hidden_dim: int,
                 pyramid_level_count: int,
                 anchor_normalized_base_size: float = 0.05,
                 anchor_logit_eps: float = 1e-2,
                 anchor_top_k: int = 300,
                 precalculate_anchor_geometry_logits: bool = True):
        super().__init__()

        # ---------------------------------------------------------------------
        # QUERY PREDECODING
        # ---------------------------------------------------------------------
        #
        # The authors feels the need to add another feature projectors.
        #
        # TODO: is it absolutely necessary from a performance point of view?
        self.feature_projectors = FeaturePyramidProjection(
            [encoding_dim] * pyramid_level_count,
            hidden_dim
        )
        self.anchor_decoder = AnchorDecoder(
            encoding_dim,
            hidden_dim,
            num_classes,
            geometry_head_layer_count=3,
            geometry_head_activation='relu',
            normalized_base_size=anchor_normalized_base_size,
            logit_eps=anchor_logit_eps,
            precalculate_anchor_geometry_logits=precalculate_anchor_geometry_logits
        )

        self.anchor_selector = AnchorSelector(anchor_top_k)

        assert pyramid_level_count == 3
        self.decoder = MultiScaleDeformableTransformerDecoder(
            32, [4, 4, 4],
            num_classes=num_classes,
            attn_head_count=8,
            attn_feedforward_dim=64,
            attn_num_layers=6,
            attn_dropout=0.1
        )

    def _transform_feature_pyramid_into_memory(
        self,
        feature_pyramid: list[torch.Tensor]
     ) -> torch.Tensor:
        """Fuse the feature maps of the feature pyramid into a single memory
        matrix.

        The memory matrix is basically the value matrix that will be used for
        attention-based decoder.
        """
        object_query_matrices = [
            fmap.flatten(2).permute(0, 2, 1)
            for fmap in feature_pyramid
        ]
        object_query_matrix_final = torch.cat(object_query_matrices, dim=1)
        return object_query_matrix_final

    def forward(
        self,
        feature_pyramid: list[torch.Tensor]
    ):
        fpyr_projected = self.feature_projectors(feature_pyramid)
        fpyr_sizes = [
            # Extract (w, h) from the shape (n, c, h, w)
            (fmap.shape[3], fmap.shape[2])
            for fmap in fpyr_projected
        ]

        memory = self._transform_feature_pyramid_into_memory(
            fpyr_projected
        )

        (memory_projected,
         anchor_class_logits,
         anchor_geometry_logits) = self.anchor_decoder.forward(
             memory,
             fpyr_sizes
         )

        # Sort the queries by decreasing logit values and keep the top-K
        # queries.
        (top_queries,
         top_class_logits,
         top_geom_logits) = self.anchor_selector.forward(
             memory_projected,
             anchor_class_logits,
             anchor_geometry_logits
         )
