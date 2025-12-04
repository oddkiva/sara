from typing import Iterable

from oddkiva.brahma.torch.object_detection.detr.architectures.rtdetr.encoder.hybrid_encoder import FeaturePyramidProjection
import torch
import torch.nn as nn

from oddkiva.brahma.torch.backbone.resnet.rtdetrv2_variant import (
    UnbiasedConvBNA
)
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.anchor_decoder import AnchorGeometryLogitEnumerator
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.checkpoint import AnchorDecoder
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.anchor_selector import AnchorSelector


class QuerySelector(nn.Module):
    """
    This is the so-called Uncertainty-Minimal Query Selector class as described
    in RT-DETR.
    """

    def __init__(self,
                 query_dim: int,
                 query_hidden_dim: int,
                 query_num_classes: int,
                 query_pyramid_wh_sizes: list[tuple[int, int]],
                 top_K: int,
                 geometry_head_layer_count: int = 3,
                 geometry_head_activation: str = 'relu',
                 initial_object_class_probability: float = 0.1,
                 precalculate_anchor_geometry_logits: bool = True,
                 anchor_normalized_base_size: float = 0.05,
                 anchor_logit_eps: float = 0.01):
        super().__init__()

        # ---------------------------------------------------------------------
        # QUERY PREDECODING
        # ---------------------------------------------------------------------
        #
        # The authors feels the need to add another feature projectors.
        #
        # TODO: is it absolutely necessary from a performance point of view?
        pyramid_level_count = len(query_pyramid_wh_sizes)
        self.feature_projectors = FeaturePyramidProjection(
            [encoding_dim] * pyramid_level_count,
            hidden_dim
        )

        self.anchor_decoder = AnchorDecoder(
            query_dim,
            query_hidden_dim,
            query_num_classes,
            geometry_head_layer_count=geometry_head_layer_count,
            geometry_head_activation=geometry_head_activation,
            normalized_base_size=anchor_normalized_base_size,
            logit_eps=anchor_logit_eps,
            precalculate_anchor_geometry_logits=precalculate_anchor_geometry_logits,
            image_pyramid_wh_sizes=query_pyramid_wh_sizes,
            initial_class_probability=initial_object_class_probability
        )
        self.anchor_selector = AnchorSelector(top_k=top_K)


        self._reinitialize_learning_parameters()

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

    def forward(self, feature_pyramid: list[torch.Tensor]):
        feature_pyramid_projected = [
            proj(fmap)
            for proj, fmap in zip(self.feature_projectors, feature_pyramid)
        ]

        feature_pyramid_sizes = [
            # Extract (w, h) from the shape (n, c, h, w)
            (fmap.shape[3], fmap.shape[2])
            for fmap in feature_pyramid_projected
        ]

        memory = self._transform_feature_pyramid_into_memory(
            feature_pyramid_projected
        )

        (memory_postprocessed,
         anchor_class_logits,
         anchor_geometry_logits) = self.anchor_decoder.forward(
             memory,
             feature_pyramid_sizes
         )

        # Sort the queries by decreasing logit values and keep the top 300
        # queries.

        return (memory_postprocessed,
                anchor_class_logits,
                anchor_geometry_logits)
