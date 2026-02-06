from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.config import RTDETRConfig


class RTDETRv2(nn.Module):

    def __init__(self, config: RTDETRConfig):
        super().__init__()

        self.backbone = config.backbone.make_model()
        self.encoder = config.encoder.make_model()
        self.query_selector = config.query_selector.make_model()
        self.decoder = config.decoder.make_model()

        level_count = len(self.encoder.backbone_feature_proj.projections)
        assert level_count == len(
            self.query_selector.feature_projectors.projections
        )
        self.pyramid_level_count = level_count

    def forward(
        self,
        x: torch.Tensor,
        targets: dict[str, list[torch.Tensor]] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        # ResNet-like backbone.
        feature_pyramid = self.backbone(x)

        # AIFI+CCFF
        encoding_pyramid = self.encoder.forward(
            feature_pyramid[-self.pyramid_level_count:]
        )

        # Top-K anchor selection.
        (top_anchor_queries,
         top_anchor_class_logits,
         top_anchor_geometry_logits,
         memory) = self.query_selector.forward(encoding_pyramid)

        # The value and its original feature pyramid shapes.
        value = memory
        value_mask = None
        value_pyramid_hw_sizes = [
            encoding_map.shape[2:]
            for encoding_map in encoding_pyramid
        ]

        # NOTE:
        # At inference time, ensure that the embed vectors and geometry logits
        # are made non differentiable at the decoding stage.
        (detection_boxes, detection_class_logits,
         dn_boxes, dn_class_logits,
         dn_groups) = self.decoder.forward(
             top_anchor_queries.detach(), top_anchor_geometry_logits.detach(),
             value, value_pyramid_hw_sizes,
             value_mask=value_mask,
             targets=targets
         )

        top_anchor_geometries = F.sigmoid(top_anchor_geometry_logits)

        aux_train_outputs = {
            # To optimize:
            # - the backbone,
            # - AIFI+CCFF hybrid encoder,
            # - anchor decoder inside the self.query_selector
            # The following outputs are used as feedback:
            'anchors': (top_anchor_geometries, top_anchor_class_logits),

            # Denoising groups to accelerate the training convergence.
            'dn_boxes': (dn_boxes, dn_class_logits),

            # The input denoising groups of boxes.
            'dn_groups': dn_groups
        }

        return detection_boxes, detection_class_logits, aux_train_outputs

    def backbone_learnable_params(
        self,
        name_filter: list[str] = ['batch_norm'],
        debug: bool = False
    ) -> dict[str, nn.Parameter]:
        # Collect the backbone learnable parameters.
        params_filtered = {}

        for param_name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if not param_name.startswith('backbone'):
                continue
            if any([word in param_name for word in name_filter]):
                continue
            params_filtered[param_name] = param

            if debug:
                print(f'[backbone] {param_name}: {param.shape}')

        return params_filtered

    def query_selector_learnable_params(
        self,
        name_filter: list[str] = ['batch_norm', 'layer_norm'],
        debug: bool = False
    ) -> dict[str, nn.Parameter]:
        params_filtered = {}

        for param_name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if not param_name.startswith('query_selector'):
                continue
            if any([word in param_name for word in name_filter]):
                continue
            params_filtered[param_name] = param

            if debug:
                print(f'[query-selector] {param_name}: {param.shape}')

        return params_filtered

    def encoder_learnable_params(
        self,
        name_filter: list[str] = ['batch_norm', 'layer_norm'],
        debug: bool = False
    ) -> dict[str, nn.Parameter]:
        params_filtered = {}
        for param_name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if not param_name.startswith('encoder'):
                continue
            if any([word in param_name for word in name_filter]):
                continue
            params_filtered[param_name] = param

            if debug:
                print(f'[transformer-encoder] {param_name}: {param.shape}')
        return params_filtered

    def decoder_learnable_params(
        self,
        name_filter: list[str] = ['batch_norm', 'layer_norm'],
        debug: bool = False
    ) -> dict[str, nn.Parameter]:
        params_filtered = {}

        for param_name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if not param_name.startswith('decoder'):
                continue
            if any([word in param_name for word in name_filter]):
                continue
            params_filtered[param_name] = param

            if debug:
                print(f'[transformer-decoder] {param_name}: {param.shape}')

        return params_filtered

    def group_learnable_parameters(
        self,
        backbone_filter: list[str] = ['batch_norm'],
        query_selector_filter: list[str] = ['batch_norm', 'layer_norm'],
        encoder_filter: list[str] = ['batch_norm', 'layer_norm'],
        decoder_filter: list[str] = ['batch_norm', 'layer_norm'],
        debug: bool = False
    ) -> list[dict[str, Any]]:
        """
        Populates the learnable parameter groups with the learning
        parameter default values as set in RT-DETR's original implementation.

        RT-DETR splits the parameters in 3 groups.

        1. backbone parameters excluding batch norm parameters,
        2. encoder and decoder parameters excluding batch norm and layer norm
           parameters,
        3. the rest are essentially batch norm and layer norms

        The backbone parameters should be updated with a much lower learning
        rate (1e-5), especially if we load the model weights from a pretrained
        backbone.

        Second, the encoder and decoder parameters are updated with a higher
        learning rate (1e-4) and a zero weight decay for the AdamW optimizer.

        Finally, the remaining parameters are essentially batch norm and layer
        norm parameters.

        Here, the implementation chooses to split RT-DETR v2's transformer
        decoder into 2 parts, which I refer as:

        - the query selector which fulfils the following operations:

          1. projects the multi-scale anchor features produced by the hybrid
             encoder (AIFI+CCFF) into the same feature dimension.
          2. ranks and selects these projected features according to some
             object class scoring function.

        - the multiscale deformable transformer decoder, that refines the
          anchor queries into final box geometries and object probabilities.
        """

        b_params = self.backbone_learnable_params(backbone_filter, debug)
        e_params = self.encoder_learnable_params(encoder_filter, debug)
        qs_params = self.query_selector_learnable_params(query_selector_filter,
                                                         debug)
        d_params = self.decoder_learnable_params(decoder_filter, debug)

        selected_params = {
            **b_params,
            **e_params,
            **qs_params,
            **d_params,
        }

        remaining_params = {}
        for param_name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param_name in selected_params:
                continue
            remaining_params[param_name] = param

            if debug:
                print(f'[rt-detr v2 remaining] {param_name}: {param.shape}')

        # TODO: improve this. I am not entirely satisfied...
        #
        # Set the learning parameters with default values and the user can
        # modify later them at his/her convenience.
        return [
            # Backbone.
            {
                'params': [p for _, p in b_params.items()],
                'lr': 1e-5,
            },
            # Hybrid encoder (AIFI+CCFF).
            {
                'params': [p for _, p in e_params.items()],
                'lr': 1e-4,
                'weight_decay': 0
            },
            # Query selector.
            {
                'params': [p for _, p in qs_params.items()],
                'lr': 1e-4,
                'weight_decay': 0,
            },
            # Multi-scale deformable transformer decoder.
            {
                'params': [p for _, p in d_params.items()],
                'lr': 1e-4,
                'weight_decay': 0,
            },
            # Remaining parameters (basically all the batch norms and layer
            # norms)
            {
                'params': [p for _, p in remaining_params.items()],
                'lr': 1e-4,
            },
        ]
