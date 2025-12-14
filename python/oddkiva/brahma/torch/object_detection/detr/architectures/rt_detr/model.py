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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature_pyramid = self.backbone(x)
        encoding_pyramid = self.encoder(
            feature_pyramid[-self.pyramid_level_count:]
        )

        (top_queries,
         top_class_logits,
         top_geometry_logits,
         memory) = self.query_selector(encoding_pyramid)

        value = memory
        value_mask = None
        value_pyramid_hw_sizes = [
            encoding_map.shape[2:]
            for encoding_map in encoding_pyramid
        ]

        # IMPORTANT: ensure the embed vectors and geometry logits are made non
        # differentiable at the decoding stage.
        (detection_boxes, detection_class_logits,
         dn_boxes, dn_class_logits) = self.decoder.forward(
             top_queries.detach(), top_geometry_logits.detach(),
             value, value_pyramid_hw_sizes,
             value_mask=value_mask,
             targets=targets
         )

        train_outputs = (
            # To optimize:
            # - the backbone,
            # - AIFI+CCFF hybrid encoder,
            # - anchor decoder inside the self.query_selector
            # The following outputs are used as feedback:
            top_class_logits,
            top_geometry_logits,

            # To optimize the decoder:
            # - Each detection boxes and class logits are tensors that contains
            detection_boxes, detection_class_logits,
            # To accelerate convergence:
            dn_boxes, dn_class_logits

        )

        return detection_boxes, detection_class_logits
