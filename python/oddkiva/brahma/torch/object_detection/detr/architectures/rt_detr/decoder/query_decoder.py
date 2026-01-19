# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from collections import OrderedDict

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from oddkiva.brahma.torch.backbone.multi_layer_perceptron import (
    MultiLayerPerceptron
)
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    dn_detr.contrastive_denoising_group_generator import (
        ContrastiveDenoisingGroupGenerator,
        inverse_sigmoid
    )
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    deformable_detr.multiscale_deformable_attention import (
        MultiscaleDeformableAttention
    )


class BoxClassEmbeddingMap(nn.Embedding):
    """
    This embedding also learns what the embed vector of the *non-object* class
    aka token.
    """

    def __init__(self, embed_dim: int, class_count: int):
        super().__init__(class_count + 1, embed_dim, padding_idx=class_count)
        nn.init.normal_(self.weight[:-1])


class BoxGeometryEmbeddingMap(MultiLayerPerceptron):

    def __init__(self, embed_dim: int):
        super().__init__(4, 2 * embed_dim, embed_dim, 2, activation='relu')
        self._reinitialize_learning_parameters()

    def _reinitialize_learning_parameters(self):
        for layer in self.layers:
            assert type(layer) is nn.Linear
            nn.init.xavier_uniform_(layer.weight)


class BoxObjectClassLogitHead(nn.Linear):

    def __init__(self,
                 embed_dim: int,
                 num_classes: int,
                 initial_prob: float = 0.1):
        super().__init__(embed_dim, num_classes)
        nn.init.constant_(
            self.bias,
            self._initial_class_logit_value(initial_prob)
        )

    def _initial_class_logit_value(self, prob: float) -> float:
        """
        The interpretation is the bias is a low starting probability value that
        any anchor with box size of 5% of the image size contains an object of
        any class.
        """
        logit_value = -math.log((1 - prob) / prob)
        return logit_value


class BoxGeometryLogitHead(MultiLayerPerceptron):

    def __init__(self,
                 embed_dim: int,
                 layer_count: int,
                 activation: str = 'relu'):
        super().__init__(embed_dim, embed_dim, 4, layer_count,
                         activation=activation)
        self._reinitialize_learning_parameters()

    def _reinitialize_learning_parameters(self):
        # Only reinitialize the last layer.
        nn.init.constant_(self.layers[-1].weight, 0)
        nn.init.constant_(self.layers[-1].bias, 0)


class MultiScaleDeformableTransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        value_dim: int,
        num_heads: int,
        image_level_count: int,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        normalize_before: bool = False
    ):
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

        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout,
            batch_first=True
        )

        self.dropout_1 = nn.Dropout(p=dropout)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)

        self.cross_attention = MultiscaleDeformableAttention(
            embed_dim, num_heads, value_dim, image_level_count
        )
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

        self.feedforward = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(embed_dim, feedforward_dim)),
            ("activation", nn.ReLU()),
            ("dropout", nn.Dropout(p=dropout)),
            ("linear2", nn.Linear(feedforward_dim, embed_dim))
        ]))
        assert type(self.feedforward.linear1) is nn.Linear
        assert type(self.feedforward.linear2) is nn.Linear
        self.dropout_3 = nn.Dropout(p=dropout)
        self.layer_norm_3 = nn.LayerNorm(embed_dim)

    def with_positional_embeds(
        self,
        queries: torch.Tensor,
        positional_embeds: torch.Tensor | None
    ) -> torch.Tensor:
        if positional_embeds is None:
            return queries
        else:
            return queries + positional_embeds

    def forward(
        self,
        query_embeds: torch.Tensor,
        query_geometries: torch.Tensor,
        memory: torch.Tensor,
        memory_spatial_sizes: list[tuple[int, int]],
        query_positional_embeds: torch.Tensor | None = None,
        query_self_attn_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""
        Decodes the object query matrix $\mathbf{Q}$ into an output value matrix
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
            \mathbf{Q} \left( \mathbf{F} + \phi(\mathbf{X})^\intercal \right)
        \right) \mathbf{V}
        $$

        Parameters:
            query_embeds:
                The matrix $\mathbf{Q}$, i.e., the list of query encodings
                stacked as row vectors
            query_geometries:
                The list of 4D box geometries associated to each object query
                row vectors of $\mathbf{Q}$.
            query_self_attn_mask:
                The self-attention mask.

                It is used only at training time to optimize the auxiliary
                contrastive denoising task of perturbed ground-truth labeled
                boxes.
            memory:
                the concatenated query vectors that are calculated from the
                feature pyramid
                (backbone &rarr; AIFI &rarr; CCFF &rarr; projection &rarr;
                concatenation).
            memory_mask:
                the attention mask used for the deformable cross-attention
                layer.

        Returns:
            Decoded queries $\mathbf{V}^+$
        """

        # IMPORTANT
        # ---------
        #
        # In RT-DETR, a query is considered to be a new independent input even
        # if it is actually produced by the CNN backbone extractor and the
        # encoder.
        assert query_geometries.requires_grad is False

        # Prepare the data.
        Q = K = self.with_positional_embeds(query_embeds, query_positional_embeds)
        V = query_embeds

        # 1. Self-Attention -> (Dropout -> Add -> Norm).
        #
        # 1.1. Self-attention.
        ΔV, _ = self.self_attention.forward(
            Q, K,
            value=V,
            attn_mask=query_self_attn_mask
        )
        # 1.2.a) Perturb the enhanced value residuals to avoid overfitting in the
        #        self-attention block with the dropout layer.
        # 1.2.b) Apply the (Add -> Norm) layer.
        V_super = self.layer_norm_1(V + self.dropout_1(ΔV))

        # 2. Cross-attention -> (Dropout -> Add -> Norm)
        ΔV = self.cross_attention.forward(
            self.with_positional_embeds(V_super, query_positional_embeds),
            query_geometries,
            memory,
            memory_spatial_sizes,
            value_mask=memory_mask
        )
        V_super = self.layer_norm_2(V_super + self.dropout_2(ΔV))

        # 3. FFN -> (Dropout -> Add -> Norm)
        ΔV = self.feedforward(V_super)
        V_super = self.layer_norm_3(V_super + self.dropout_3(ΔV))

        return V_super


class MultiScaleDeformableTransformerDecoder(nn.Module):
    """This class stacks a sequence of
    `MultiScaleDeformableTransformerDecoderLayer`.

    Essentially, what this does is that it iteratively refines the box geometry
    and object class probability vector for each object query.
    """

    def __init__(
        self,
        hidden_dim: int,
        value_dim: int,
        kv_count_per_level: list[int],
        num_classes: int = 80,
        # ATTENTION PARAMS
        attn_head_count: int = 8,
        attn_feedforward_dim: int = 2048,
        attn_num_layers: int = 6,
        attn_dropout: float = 0.1,
        normalize_before: bool = False,
    ):
        super().__init__()

        # ---------------------------------------------------------------------
        # OBJECT QUERY DECODING
        # ---------------------------------------------------------------------
        self.layers = nn.ModuleList([
            MultiScaleDeformableTransformerDecoderLayer(
                hidden_dim, value_dim, attn_head_count,
                len(kv_count_per_level),
                feedforward_dim=attn_feedforward_dim,
                dropout=attn_dropout,
                normalize_before=normalize_before
            )
            for _ in range(attn_num_layers)
        ])

        # These two heads are mainly there to accelerate the training
        # convergence
        #
        # Query class logits
        self.decoder_class_logits_head = nn.ModuleList(
            nn.Linear(hidden_dim, num_classes) for _ in range(attn_num_layers)
        )
        # Query geometry logits
        self.decoder_box_geometry_head = nn.ModuleList(
            MultiLayerPerceptron(hidden_dim, hidden_dim, 4, 3)
            for _ in range(attn_num_layers)
        )

        # ---------------------------------------------------------------------
        # OBJECT QUERY RE-EMBEDDING FUNCTIONS
        # ---------------------------------------------------------------------
        # The two decoupled inverse functions that we learn in order to
        # transform:
        # - the box labels into a hidden object class embedding space
        # - the box geometries into a hidden object geometry embedding space
        #
        # These are encoding that are reused for the transformer decoder.
        self.box_class_embedding_map = BoxClassEmbeddingMap(
            hidden_dim,
            num_classes
        )
        self.box_geometry_embedding_map = BoxGeometryEmbeddingMap(hidden_dim)

        self.dn_group_gen = ContrastiveDenoisingGroupGenerator(num_classes)

        # Auxiliary geometry estimator for each decoding iteration.
        self.box_geometry_logit_heads = nn.ModuleList(
            BoxGeometryLogitHead(hidden_dim, 3, activation='relu')
            for _ in range(attn_num_layers)
        )
        # Auxiliary object class estimator for each decoding iteration.
        self.box_class_logit_heads = nn.ModuleList(
            BoxObjectClassLogitHead(hidden_dim, num_classes)
            for _ in range(attn_num_layers)
        )

    def decode(
        self,
        query: torch.Tensor, query_geometry_logits: torch.Tensor,
        value: torch.Tensor,
        value_spatial_sizes: list[tuple[int, int]],
        query_self_attn_mask: torch.Tensor | None = None,
        value_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE
        # The commented code is valid only at inference time and not
        # during training time, because we generate denoising groups of queries
        # by embedding space, space of which we are trying to learn.
        #
        # assert query.requires_grad is False
        assert query_geometry_logits.requires_grad is False

        # Get the actual query geometry by activating the logits with the
        # sigmoid function.

        # Initialize:
        # - the current query embedding
        # - the current query geometry logits (the object geometry before the
        #   sigmoid activation)
        # - the current query geometry (the object box geometry)
        query_curr = query
        query_geom_logits_curr = query_geometry_logits
        query_geom_curr = F.sigmoid(query_geom_logits_curr)

        query_next: torch.Tensor | None = None
        query_class_logits_next: torch.Tensor | None = None
        query_geom_logits_next: torch.Tensor | None = None
        query_geom_next: torch.Tensor | None = None

        query_geometries_denoised = []
        query_class_logits_denoised = []

        for i, decoder_layer in enumerate(self.layers):
            assert type(decoder_layer) is \
                MultiScaleDeformableTransformerDecoderLayer

            # Calculate the corresponding embed vector for each box geometry
            query_geom_embed_curr = self.box_geometry_embedding_map\
                .forward(query_geom_curr)

            # Denoise the current query.
            query_next = decoder_layer.forward(
                query_curr, query_geom_curr,
                value, value_spatial_sizes,
                query_positional_embeds=query_geom_embed_curr,
                query_self_attn_mask=query_self_attn_mask,
                memory_mask=value_mask
            )

            # Estimate the new object class logits (object class probabilities).
            query_class_logits_next = self.box_class_logit_heads[i](query_next)

            # Estimate the new object geometry (cx, cy, w, h).
            Δ_query_geom_logits = self.box_geometry_logit_heads[i](query_next)
            query_geom_logits_next = \
                query_geom_logits_curr + Δ_query_geom_logits
            #   ^                        ^
            #   |                        |
            #   |-- non-diff             |-- diff
            #
            # NOTE: just like a Taylor expansion, we only care about estimating
            # the residual. Only the Δ is differentiable.
            query_geom_next = F.sigmoid(query_geom_logits_next)

            # Store the denoised results.
            query_geometries_denoised.append(query_geom_next)
            query_class_logits_denoised.append(query_class_logits_next)

            # Update for the next denoising iteration.
            query_curr = query_next
            query_geom_logits_curr = query_geom_logits_next
            # Make sure that we only optimize the residuals at the training
            # stage.
            #
            # We detach `query_geom_curr` from the gradient flow graph, to make
            # sure we don't backpropagate the gradients further to previous
            # decoding layers that calculated the previous iterations of the
            # query objects.
            query_geom_curr = query_geom_next.detach()

        return (torch.stack(query_geometries_denoised),
                torch.stack(query_class_logits_denoised))

    def augment_query_with_dn_groups(
        self,
        query: torch.Tensor,
        query_geometry_logits: torch.Tensor,
        targets: dict[str, list[torch.Tensor]]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        ContrastiveDenoisingGroupGenerator.Output
    ]:
        # Pre-conditions:
        assert query.requires_grad is False
        assert query_geometry_logits.requires_grad is False

        query_count = query.shape[1]
        dn_groups = self.dn_group_gen.forward(query_count,
                                              targets['boxes'],
                                              targets['labels'])

        if dn_groups.labels is None:
            assert dn_groups.geometries is None
            assert dn_groups.attention_mask is None
            assert dn_groups.positive_indices is None
            assert dn_groups.group_count is None
            assert dn_groups.partition is None

            return query, query_geometry_logits, dn_groups

        # `dn_groups.attention_mask` is a self-attention mask
        #
        # The denoising group generator is so-called "contrastive" as it
        # generates negative and positive samples.
        #
        # It is worth reflecting about the denoising self-attention mask is
        # doing:
        #
        # The self-attention mask reinforces the common features between
        # negative samples and likewise the common features between
        # positive samples.

        dn_query = self.box_class_embedding_map(dn_groups.labels)

        # Fuse the noised ground-truth queries and the top-K queries in
        # that following order because of the construction of the
        # self-attention mask.
        query = torch.cat(
            (dn_query, query),
            dim=1
        )
        assert dn_groups.geometries is not None
        dn_geometry_logits = inverse_sigmoid(dn_groups.geometries)
        query_geometry_logits = torch.cat(
            (dn_geometry_logits, query_geometry_logits),
            dim=1
        )

        return query, query_geometry_logits, dn_groups

    def forward(
        self,
        query: torch.Tensor,
        query_geometry_logits: torch.Tensor,
        value: torch.Tensor,
        value_spatial_sizes: list[tuple[int, int]],
        value_mask: torch.Tensor | None = None,
        targets: dict[str, list[torch.Tensor]] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor,
               torch.Tensor | None, torch.Tensor | None,
               ContrastiveDenoisingGroupGenerator.Output]:
        if targets is not None:
            (query,
             query_geometry_logits,
             dn_groups) = self.augment_query_with_dn_groups(query,
                                                            query_geometry_logits,
                                                            targets)
        else:
            dn_groups = ContrastiveDenoisingGroupGenerator.Output(
                None, None, None
            )

        query_geometries, query_class_logits = self.decode(
            query,
            query_geometry_logits,
            value, value_spatial_sizes,
            query_self_attn_mask=dn_groups.attention_mask,
            value_mask=value_mask
        )

        # Separate the denoising groups and the top-K image-based queries.
        #
        # Be careful: sometimes the data augmentation will yield no
        # ground-truth data. So the condition `if targets is not None` is not
        # sufficient
        if dn_groups.partition is not None:
            partition = dn_groups.partition
            dn_boxes, detection_boxes = torch.split(
                query_geometries, partition, dim=2
            )
            dn_class_logits, detection_class_logits = torch.split(
                query_class_logits, partition, dim=2
            )
        else:
            dn_boxes, detection_boxes = None, query_geometries
            dn_class_logits, detection_class_logits = None, query_class_logits


        return (
            detection_boxes, detection_class_logits,
            dn_boxes, dn_class_logits,
            dn_groups
        )
