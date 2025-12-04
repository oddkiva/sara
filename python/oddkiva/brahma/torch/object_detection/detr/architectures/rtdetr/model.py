import torch
import torch.nn as nn

from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.config import RTDETRConfig


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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_pyramid = self.backbone(x)
        encoding_pyramid = self.encoder(
            feature_pyramid[-self.pyramid_level_count:]
        )

        (top_queries,
         _,
         top_geometry_logits,
         memory) = self.query_selector(encoding_pyramid)

        value = memory
        value_mask = None
        value_pyramid_hw_sizes = [
            encoding_map.shape[2:]
            for encoding_map in encoding_pyramid
        ]

        box_geometries, box_class_logits = self.decoder.forward(
            top_queries.detach(), top_geometry_logits.detach(),
            value, value_pyramid_hw_sizes,
            value_mask=value_mask
        )

        return box_geometries, box_class_logits
