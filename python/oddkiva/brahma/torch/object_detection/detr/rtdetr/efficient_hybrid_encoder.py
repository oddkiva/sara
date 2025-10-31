import torch

from oddkiva.brahma.torch.object_detection.detr.rtdetr.aifi import AIFI
from oddkiva.brahma.torch.object_detection.detr.rtdetr.ccff import CCFF


class EfficientHybridEncoder(torch.nn.Module):

    def __init__(self, feature_dims: list[int],
                 attn_head_count: int,
                 attn_feedforward_dim: int = 2048,
                 attn_dropout: float = 0.1,
                 attn_num_layers: int = 6,
                 hidden_dim: int = 256):
        super().__init__()
        self.aifi = AIFI(feature_dims[-1],
                         attn_head_count,
                         feedforward_dim=attn_feedforward_dim,
                         dropout=attn_dropout,
                         num_layers=attn_num_layers)
        self.ccff = CCFF(feature_dims, hidden_dim)

    def forward(self, feature_pyramid: list[torch.Tensor]) -> torch.Tensor:
        F5 = self.aifi.forward(feature_pyramid[-1])
        S = feature_pyramid
        Q = self.ccff.forward(F5, S)
        return Q
