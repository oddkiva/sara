import torch


class EfficientHybridEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, feature_pyramid: Iterable[torch.Tensor]) -> torch.Tensor:
        pass
