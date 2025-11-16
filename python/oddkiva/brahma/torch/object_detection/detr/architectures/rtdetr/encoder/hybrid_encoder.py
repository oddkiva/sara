import torch

from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.encoder.feature_pyramid_projection import FeaturePyramidProjection
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.encoder.aifi import AIFI
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.encoder.ccff import CCFF


class HybridEncoder(torch.nn.Module):
    """
    1. For each feature map, we project each feature vector into a smaller
       feature space of dimension `hidden_dim`
    2. Apply the self-attention transformer encoder `AIFI` to the coarsest
       feature map.
    3. Inject the semantic information top-down and bottom-up with `CCFF`.
    """

    def __init__(self,
                 input_feature_dims: list[int],
                 attn_head_count: int,
                 hidden_dim: int = 256,
                 attn_feedforward_dim: int = 2048,
                 attn_dropout: float = 0.1,
                 attn_num_layers: int = 6):
        super().__init__()

        self.backbone_feature_proj = FeaturePyramidProjection(
            input_feature_dims, hidden_dim)
        self.aifi = AIFI(hidden_dim,
                         attn_head_count,
                         feedforward_dim=attn_feedforward_dim,
                         dropout=attn_dropout,
                         num_layers=attn_num_layers)
        self.ccff = CCFF(len(input_feature_dims) - 1, hidden_dim)

    def forward(
        self,
        feature_pyramid: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        # Project the feature vectors of the feature pyramid into the same
        # dimensional space.
        S = self.backbone_feature_proj(feature_pyramid)
        # Perform self-attention of the coarsest feature map of the feature
        # pyramid.
        # [F3, F4, F5] for ResNet
        F5 = self.aifi.forward(S[-1])
        # The top-down then bottom-up fusion scheme.
        Q = self.ccff.forward(F5, S)
        return Q
