import torch
import torch.nn as nn

from oddkiva.brahma.torch.backbone.resnet.rtdetrv2_variant import (
    ResNet50RTDETRV2Variant
)
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.encoder.hybrid_encoder import HybridEncoder
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.query_selector import QuerySelector
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.checkpoint import MultiScaleDeformableTransformerDecoder


class RTDETRv2(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = ResNet50RTDETRV2Variant()
        self.encoder = HybridEncoder()
        self.query_selector = QuerySelector()
        self.decoder = MultiScaleDeformableTransformerDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_pyramid = self.backbone(x)
        anchor_query_pyramid = self.encoder(feature_pyramid)
        top_queries = self.query_selector(anchor_query_pyramid)
        queries = self.decoder(top_queries, anchor_query_pyramid)
        return queries
