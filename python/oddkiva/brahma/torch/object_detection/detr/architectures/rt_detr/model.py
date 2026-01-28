from typing import Any

import torch
import torch.nn as nn

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
        encoding_pyramid = self.encoder(
            feature_pyramid[-self.pyramid_level_count:]
        )

        # Top-K anchor selection.
        (top_queries,
         top_class_logits,
         top_geometry_logits,
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
             top_queries.detach(), top_geometry_logits.detach(),
             value, value_pyramid_hw_sizes,
             value_mask=value_mask,
             targets=targets
         )

        aux_train_outputs = {
            # To optimize:
            # - the backbone,
            # - AIFI+CCFF hybrid encoder,
            # - anchor decoder inside the self.query_selector
            # The following outputs are used as feedback:
            'top_k_anchor_boxes': (top_geometry_logits, top_class_logits),

            # Denoising groups to accelerate the training convergence.
            'dn_boxes': (dn_boxes, dn_class_logits),

            # The input denoising groups of boxes.
            'dn_groups': dn_groups
        }

        return detection_boxes, detection_class_logits, aux_train_outputs

    def backbone_learnable_params(
        self,
        name_filter: list[str] = ['batch_norm']
    ) -> dict[str, nn.Parameter]:
        # Collect the backbone learnable parameters.
        params_filtered = {}

        for param_name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' not in param_name:
                continue
            if any([word in param_name for word in name_filter]):
                continue
            params_filtered[param_name] = param
            print(f'[backbone] {param_name}: {param.shape}')

        return params_filtered

    def query_selector_learnable_params(
        self,
        name_filter: list[str] = ['batch_norm', 'layer_norm']
    ) -> dict[str, nn.Parameter]:
        params_filtered = {}

        for param_name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'query_selector' not in param_name:
                continue
            if any([word in param_name for word in name_filter]):
                continue
            params_filtered[param_name] = param
            print(f'[query-selector] {param_name}: {param.shape}')

        return params_filtered

    def encoder_learnable_params(
        self,
        name_filter: list[str] = ['batch_norm', 'layer_norm']
    ) -> dict[str, nn.Parameter]:
        params_filtered = {}
        for param_name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'encoder' not in param_name:
                continue
            if any([word in param_name for word in name_filter]):
                continue
            params_filtered[param_name] = param
            print(f'[transformer-encoder] {param_name}: {param.shape}')
        return params_filtered

    def decoder_learnable_params(
        self,
        name_filter: list[str] = ['batch_norm', 'layer_norm']
    ) -> dict[str, nn.Parameter]:
        params_filtered = {}

        for param_name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'decoder' not in param_name:
                continue
            if any([word in param_name for word in name_filter]):
                continue
            params_filtered[param_name] = param
            print(f'[transformer-decoder] {param_name}: {param.shape}')

        return params_filtered

    def populate_learnable_parameter_groups(
        self,
        backbone_filter: list[str] = ['batch_norm'],
        query_selector_filter: list[str] = ['batch_norm', 'layer_norm'],
        encoder_filter: list[str] = ['batch_norm', 'layer_norm'],
        decoder_filter: list[str] = ['batch_norm', 'layer_norm'],
    ) -> list[dict[str, Any]]:
        b_params = self.backbone_learnable_params(backbone_filter)
        qs_params = self.query_selector_learnable_params(query_selector_filter)
        e_params = self.encoder_learnable_params(encoder_filter)
        d_params = self.decoder_filtered_params(decoder_filter)

        remaining_params = {}
        for param_name, param in self.named_parameters():
            if param_name in b_params:
                continue
            if param_name in qs_params:
                continue
            if param_name in e_params:
                continue
            if param_name in d_params:
                continue
            remaining_params[param_name] = param
            print(f'[rt-detr v2 remaining] {param_name}: {param.shape}')


        return [
            # Backbone
            {
                'params': [p for _, p in b_params.items()],
                'lr': 1e-5,
            },
            # Encoder (AIFI+CCFF)
            {
                'params': [p for _, p in e_params.items()],
                'lr': 1e-4,
                'weight_decay': 0
            },
            # Query selector
            {
                'params': [p for _, p in qs_params.items()],
                'lr': 1e-4,
                'weight_decay': 0,
            },
            # Transformer Decoder.
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
                'betas': (0.9, 0.999),
                'weight_decay': 0.0001,
            },
        ]
