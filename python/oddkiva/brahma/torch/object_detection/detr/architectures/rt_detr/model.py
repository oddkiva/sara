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
