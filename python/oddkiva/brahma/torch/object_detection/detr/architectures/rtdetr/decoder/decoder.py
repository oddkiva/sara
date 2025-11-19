from collections import OrderedDict

import torch
import torch.nn as nn

from oddkiva.brahma.torch.backbone.multi_layer_perceptron import (
    MultiLayerPerceptron
)
from oddkiva.brahma.torch.backbone.resnet.rtdetrv2_variant import UnbiasedConvBNA
from oddkiva.brahma.torch.object_detection.common.anchors import (
    calculate_anchor_logits,
    enumerate_anchor_pyramid,
)
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.multiscale_deformable_attention import (
        MultiscaleDeformableAttention
    )


class AnchorLogitGenerator(nn.Module):
    """
    This class enumerates all possible anchor boxes at every level of an image
    pyramid.

    The geometry of an anchor is encoded as a 4D vector, which is considered to
    be an *activated* value. The activation function we use is specifically the
    sigmoid function. It is an increasing function whose inverse function can
    be calculated analytically.
    """

    def __init__(self,
                 normalized_base_box_size: float = 0.05,
                 eps: float = 1e-2):
        """Constructs an `AnchorLogitGenerator` object

        Parameters:
            normalized_base_box_size:
                base box size relative to the image sizes
            eps:
                epsilon value used to filter out corner-case logit values
        """
        super().__init__()

        self.normalized_base_box_size = normalized_base_box_size
        self.eps = eps

    @torch.no_grad()
    def forward(self,
                pyramid_image_wh_sizes: list[tuple[int, int]],
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_pyramid = enumerate_anchor_pyramid(
            pyramid_image_wh_sizes,
            normalized_base_box_size=self.normalized_base_box_size,
            normalize_anchor_geometry=True,
            device=device
        )

        anchors = torch.cat(anchor_pyramid, dim=0)
        anchor_logits, anchor_mask = calculate_anchor_logits(
            anchors, eps=self.eps
        )
        return anchor_logits, anchor_mask


class ObjectQueryGeometryResidualHead(MultiLayerPerceptron):

    def __init__(self,
                 encoding_dim: int,
                 hidden_dim: int,
                 layer_count: int,
                 activation='relu'):
        super().__init__(
            encoding_dim, hidden_dim, 4,  # (Δcx, Δcy, Δw, Δh)
            layer_count, activation=activation
        )


class ObjectQueryPredecoder(nn.Module):
    """
    After projecting the feature maps, then flattening and concatenating them
    into the so-called *memory* tensor, this module starts decoding this
    memory tensor.

    Namely, the decoder predicts the object class logits and the encoded
    geometry *residuals*.

    The object class logits will be used in another query selection module to
    rank and keep the best object queries from the memory tensor.
    """

    def __init__(self,
                 encoding_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 box_geometry_head_layer_count: int = 3,
                 box_geometry_activation: str = 'relu'):
        self._neck = nn.Sequential(OrderedDict([
            ('projector', nn.Linear(hidden_dim, hidden_dim)),
            ('layer_norm', nn.LayerNorm(hidden_dim,))
        ]))

        self._class_logit_head = nn.Linear(hidden_dim, num_classes)

        self._geometry_residual_head = ObjectQueryGeometryResidualHead(
            encoding_dim, hidden_dim, box_geometry_head_layer_count,
            activation=box_geometry_activation
        )

        self._reinitialize_learning_parameters()

    def _reinitialize_learning_parameters(self):
        nn.init.xavier_uniform_(self._neck[0].weight)

        bias = bias_init_with_prob(0.01)
        nn.init.constant_(self._class_logit_head.bias, bias)
        nn.init.constant_(self._geometry_residual_head.layers[-1].weight, 0)
        nn.init.constant_(self._geometry_residual_head.layers[-1].bias, 0)

    def forward(
        self,
        memory: torch.Tensor,
        anchors: torch.Tensor,
        anchor_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory_post = anchor_mask.to(dtype=memory.dtype) * memory
        memory_post = self._neck(memory_post)

        query_class_logits = self._class_logit_head(memory_post)
        query_geometry_residuals_unactivated = \
            self._geometry_residual_head(memory_post)

        query_geometries_unactivated = \
            anchors + query_geometry_residuals_unactivated

        return query_class_logits, query_geometries_unactivated


class ObjectQuerySelector(nn.Module):

    def __init__(self, top_k: int = 300):
        super().__init__()
        self._top_k = top_k

    def forward(
        self,
        memory: torch.Tensor,
        query_class_logits: torch.Tensor,
        query_geometries_unactivated: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, topk_ind = torch.topk(
            query_class_logits.max(-1).values,
            self._topk,
            dim=-1
        )

        topk_ind: torch.Tensor

        topk_coords = query_geometries_unactivated.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1,
                                                query_geometries_unactivated.shape[-1])
        )

        topk_logits = query_class_logits.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1,
                                                query_class_logits.shape[-1])
        )

        topk_memory = memory.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1])
        )

        return topk_memory, topk_logits, topk_coords


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


class MultiScaleDeformableTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 image_level_count: int,
                 feedforward_dim: int = 2048,
                 dropout: float = 0.1,
                 normalize_before: bool = False):
        """Constructs the base layer of a Transformer Decoder block with
        reasonable default parameters.

        Parameters:
            embed_dim:
                The output embedding feature dimension.
            num_heads:
                The number of attention heads in each decoder layer.
            feedforward_dim:
                The output feature dimension of the feed-forward network.
            dropout:
                The dropout probability value.
            normalize_before:
                Optionally choose to apply the layer norm operation:

                1. *before* applying the attention layer, and
                2. *before* applying the feed-forward network.

                By default, the layer norm operation is applied:

                1. *after* the attention layer, and
                2. *after* the feed-forward network.
        """
        super().__init__()

        self.normalize_before = normalize_before

        self.self_attention = torch.nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout,
            batch_first=True
        )

        self.dropout_1 = torch.nn.Dropout(p=dropout)
        self.layer_norm_1 = torch.nn.LayerNorm(embed_dim)

        self.cross_attention = MultiscaleDeformableAttention(
            embed_dim, num_heads, embed_dim, image_level_count
        )
        self.dropout_2 = torch.nn.Dropout(p=dropout)
        self.layer_norm_2 = torch.nn.LayerNorm(embed_dim)

        self.feedforward = torch.nn.Sequential(OrderedDict([
            ("linear-1", torch.nn.Linear(embed_dim, feedforward_dim)),
            ("activation", torch.nn.ReLU()),
            ("dropout", torch.nn.Dropout(p=dropout)),
            ("linear-2", torch.nn.Linear(feedforward_dim, embed_dim))
        ]))
        self.dropout_3 = torch.nn.Dropout(p=dropout)
        self.layer_norm_3 = torch.nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        memory: list[torch.Tensor]
    ) -> torch.Tensor:
        r"""Decodes the query matrix $\mathbf{Q}$ into an output value matrix
        $\mathbf{V}^{+}$.

        - The memory is a peculiar term that simply denotes the 2D feature maps
          $\mathbf{F}$, which, when flattened, corresponds to the value matrix
          $\mathbf{V}$.

        - Denoting the position of the query vector $\mathbf{q}$ by $\mathbf{x}$,
          we calculate its positional embedding $\phi(\mathbf{x})$ and we
          obtain the key matrix $\mathbf{K} = \mathbf{F} + \phi(\mathbf{X})$.

        In the end, we calculate the decoded object queries as

        $$
        \mathbf{V}^+ = \sigma \left( \frac{1}{\sqrt{d_k}}
            \mathbf{Q} (\mathbf{F} + \phi(\mathbf{X})^\intercal
        \right) \mathbf{V}
        $$


        Parameters:
            query:
                The query matrix $\mathbf{Q}$
            query_pos:
                The 2D coordinates of each query vectors of the matrix
                $\mathbf{Q}$.
            memory:
                the feature maps.

        Returns:
            Decoded queries $\mathbf{V}^+$
        """
        pass


class MultiScaleDeformableTransformerDecoder(torch.nn.Module):

    def __init__(
        self,
        encoding_dim: int,
        hidden_dim: int,
        kv_count_per_level: list[int],
        # QUERY SELECTION
        num_classes: int = 80,
        # ATTENTION PARAMS
        attn_head_count: int = 8,
        attn_feedforward_dim: int = 2048,
        attn_num_layers: int = 6,
        attn_dropout: float = 0.1,
        normalize_before: bool = False,
    ):
        super().__init__()

        # Extra parameters.
        self.anchor_logits_generator = AnchorLogitGenerator(
            normalized_base_box_size=0.05,
            eps = 1e-2
        )

        # ---------------------------------------------------------------------
        # QUERY SELECTION
        # ---------------------------------------------------------------------
        #
        # The authors feels the need to add another feature projectors.
        #
        # TODO: is it absolutely necessary from a performance point of view?
        self.feature_projectors = torch.nn.ModuleList([
            UnbiasedConvBNA(encoding_dim, hidden_dim, 1, 1, activation=None)
            for _ in range(len(kv_count_per_level))
        ])

        # ---------------------------------------------------------------------
        # OBJECT QUERY DECODING
        # ---------------------------------------------------------------------
        self.decoder = nn.ModuleList([
            MultiScaleDeformableTransformerDecoderLayer(
                hidden_dim, attn_head_count, len(kv_count_per_level),
                attn_feedforward_dim, attn_dropout,
                normalize_before=normalize_before
            )
            for _ in range(attn_num_layers)
        ])

        # OBJECT QUERY CLASS LOGITS
        self.decoder_class_logits_head = nn.ModuleList(
            nn.Linear(hidden_dim, num_classes) for _ in range(attn_num_layers)
        )
        self.decoder_box_geometry_head = nn.ModuleList(
            MultiLayerPerceptron(hidden_dim, hidden_dim, 4, 3)
            for _ in range(attn_num_layers)
        )

    # def _reinitialize_learning_parameters(self):
    #     if self.learn_query_content:
    #         init.xavier_uniform_(self.tgt_embed.weight)

    #     for m in self.input_proj:
    #         init.xavier_uniform_(m[0].weight)

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


    def forward(self, feature_pyramid: list[torch.Tensor]) -> list[torch.Tensor]:
        feature_pyramid_projected = [
            proj(fmap)
            for proj, fmap in zip(self.feature_projectors, feature_pyramid)
        ]

        feature_pyramid_sizes = [
            # Extract (w, h) from the shape (n, c, h, w)
            (fmap.shape[3], fmap.shape[2])
            for fmap in feature_pyramid_projected
        ]

        anchor_logits, anchor_mask = self.anchor_logit_generator(
            feature_pyramid_sizes,
            feature_pyramid[0].device
        )

        memory = self._transform_feature_pyramid_into_memory(
            feature_pyramid_projected
        )

        # Sort out and keep the top 300.


        spatial_shapes = [
            fmap.shape[2:]
            for fmap in feature_pyramid_projected
        ]
