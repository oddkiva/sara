import torch
import torch.nn as nn

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

    def __init__(self):
        super().__init__()

        self.anchor_enumerator = AnchorGeometryLogitEnumerator()
        self.anchor_decoder = AnchorDecoder()
        self.anchor_selector = AnchorSelector()

        # ---------------------------------------------------------------------
        # QUERY PREDECODING
        # ---------------------------------------------------------------------
        #
        # The authors feels the need to add another feature projectors.
        #
        # TODO: is it absolutely necessary from a performance point of view?
        self.feature_projectors = torch.nn.ModuleList(
            UnbiasedConvBNA(encoding_dim, hidden_dim, 1, 1, activation=None)
            for _ in range(pyramid_level_count)
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

        self._reinitialize_learning_parameters()

    def _reinitialize_learning_parameters(self):
        # if self.learn_query_content:
        #     nn.init.xavier_uniform_(self.tgt_embed.weight)

        # Reset the parameters
        for convbna in self.feature_projectors:
            assert type(convbna) is UnbiasedConvBNA
            conv = convbna.layers[0]
            assert type(conv) is nn.Conv2d
            nn.init.xavier_uniform_(conv.weight)

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
