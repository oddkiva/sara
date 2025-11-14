import torch

from oddkiva.brahma.torch.backbone.resnet.rtdetrv2_variant import (
    UnbiasedConvBNA
)
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
        self.backbone_feature_proj = torch.nn.ModuleList([
            UnbiasedConvBNA(input_feature_dim, hidden_dim, 1, 1, id,
                            activation=None)
            for id, input_feature_dim in enumerate(input_feature_dims)
        ])
        self.aifi = AIFI(input_feature_dims[-1],
                         attn_head_count,
                         feedforward_dim=attn_feedforward_dim,
                         dropout=attn_dropout,
                         num_layers=attn_num_layers)
        self.ccff = CCFF(len(input_feature_dims) - 1, hidden_dim)

    def forward(
        self,
        feature_pyramid: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        S = [proj(P)
             for (proj, P) in zip(self.backbone_feature_proj, feature_pyramid)]
        F5 = self.aifi.forward(S[-1])
        Q = self.ccff.forward(F5, S)
        return Q
