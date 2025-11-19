import torch
import torch.nn as nn

from oddkiva.brahma.torch.backbone.resnet.rtdetrv2_variant import UnbiasedConvBNA
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.anchor_decoder import AnchorDecoder


class ObjectQueryDecoder(nn.Module):

    def __init__(self,
                 num_classes: int,
                 hidden_dim: int,
                 num_layers: int,
                 num_queries: int = 300):
        super().__init__()

        self.query_geometry_to_embedding_func = MultiLayerPerceptron(
            4,               # Unactivated query geoemtry dimension
            2 * hidden_dim,  # Internal hidden dimension
            hidden_dim,      # Embedding dimension
            2                # Number of layers
        )

        # Instantiate the decoder heads.
        #
        # Store the decoded values (box geometries and probabilities) at each
        # iteration.
        #
        # The rationale is that we want to optimize each iteration as much
        # as possible during training time.
        self.class_logits_head = nn.ModuleList(
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_layers)
        )
        self.box_geometry_head = nn.ModuleList(
            MultiLayerPerceptron(hidden_dim, hidden_dim, 4, 3)
            for _ in range(num_layers)
        )

    def _reinitialize_learning_parameters(self):
        if self.learn_query_content:
            nn.init.xavier_uniform_(self.query_embedding.weight)

        for _cls, _reg in zip(self.class_logits_head, self.box_geometry_head):
            nn.init.constant_(_cls.bias, bias)
            nn.init.constant_(_reg.layers[-1].weight, 0)
            nn.init.constant_(_reg.layers[-1].bias, 0)

        for i in range(self.query_geometry_to_embedding_func.layer_count):
            nn.init.xavier_uniform_(
                self.query_geometry_to_embedding_func.layers[i].weight
            )

    def forward(self, _: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class RTDETRv2Decoder(nn.Module):

    def __init__(self,
                 num_classes: int,
                 encoding_dim: int,
                 hidden_dim: int,
                 pyramid_level_count: int,
                 anchor_normalized_base_size: float = 0.05,
                 anchor_logit_eps: float = 1e-2,
                 precalculate_anchor_geometry_logits: bool = True):
        super().__init__()

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

        self._reinitialize_learning_parameters()

    def _reinitialize_learning_parameters(self):
        # if self.learn_query_content:
        #     nn.init.xavier_uniform_(self.tgt_embed.weight)

        # Reset the parametesr
        for convbna in self.feature_projectors:
            conv = convbna.layers[0]
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
