from collections import OrderedDict

import torch
import torch.nn as nn

from oddkiva.brahma.torch.backbone.resnet.vanilla import make_activation_func
from oddkiva.brahma.torch.backbone.resnet.rtdetrv2_variant import UnbiasedConvBNA
from oddkiva.brahma.torch.object_detection.common.anchors import (
    calculate_anchor_logits,
    enumerate_pyramid_anchors
)
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.multiscale_deformable_attention import (
        MultiscaleDeformableAttention
    )


class MultiLayerPerceptron(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 num_layers: int,
                 activation: str ='relu'):
        super().__init__()

        hidden_dim_seq = [hidden_dim] * (num_layers - 1)
        in_dim_seq = [in_dim] + hidden_dim_seq
        out_dim_seq = hidden_dim_seq + [out_dim]

        activation_fn = make_activation_func(activation)
        assert activation_fn is not None

        self.layers = nn.Sequential()
        for in_channels, out_channels in zip(in_dim_seq, out_dim_seq):
            self.layers.append(nn.Linear(in_channels, out_channels))
            self.layers.append(activation_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    @property
    def layer_count(self):
        return len(self.layers) // 2


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
        self.normalized_base_box_size = 0.05
        self.sigmoid_inverse_eps = 1e-2

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

        # After projecting the feature maps, then flattening and concatenating
        # them into a memory tensor, transform them again with this shallow
        # feedforward network.
        # This is for the query selection
        self.encoder_neck = nn.Sequential(OrderedDict([
            ('projector', nn.Linear(hidden_dim, hidden_dim)),
            ('layer_norm', nn.LayerNorm(hidden_dim,))
        ]))

        self.encoding_class_logits_head = nn.Linear(hidden_dim, num_classes)

        box_geometry_dimension = 4  # cx, cy, w, h
        mlp_layer_count = 3
        self.encoding_box_geometry_head = MultiLayerPerceptron(
            encoding_dim, hidden_dim, box_geometry_dimension, mlp_layer_count,
            activation='relu'
        )

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

    def _enumerate_anchors(
        self,
        feature_pyramid_image_sizes: list[tuple[int, int]],
        device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_pyramid = enumerate_pyramid_anchors(
            feature_pyramid_image_sizes,
            normalized_base_box_size=self.normalized_base_box_size,
            normalize_anchor_geometry=True,
            device=device
        )

        anchors = torch.cat(anchor_pyramid, dim=0)
        anchor_logits, anchor_mask = calculate_anchor_logits(
            anchors, eps=self.sigmoid_inverse_eps
        )
        return anchor_logits, anchor_mask

    def _estimate_obj_class_logits(self, memory: torch.Tensor) -> torch.Tensor:
        self.encoding_class_logits_head

    def _select_topk(self,
                     memory: torch.Tensor,
                     outputs_logits: torch.Tensor,
                     outputs_coords_unact: torch.Tensor,
                     topk: int):
        _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        topk_ind: torch.Tensor

        topk_coords = outputs_coords_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_coords_unact.shape[-1]))

        topk_logits = outputs_logits.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))

        topk_memory = memory.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_coords

    def feature_pyramid_as_memory_matrix(
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
            fmap.shape[2:][::-1]
            for fmap in feature_pyramid_projected
        ]

        anchor_logits, anchor_mask = self._enumerate_anchors(
            feature_pyramid_sizes,
            feature_pyramid[0].device
        )

        # Zero out infinite values and negative values
        memory = anchor_mask.to(dtype=memory.dtype) * anchor_logits
        memory  = self.encoder_neck(memory)
        obj_cls_logits = self.encoding_class_logits_head(memory)
        obj_box_geometry_unactivated = self.encoding_box_geometry_head(memory)

        # Sort out and keep the top 300.


        memory = self.feature_pyramid_as_memory_matrix(
            feature_pyramid_projected
        )
        spatial_shapes = [
            fmap.shape[2:]
            for fmap in feature_pyramid_projected
        ]
