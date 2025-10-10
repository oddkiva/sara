import torch
import torchvision


class ReidDescriptor50(torch.nn.Module):

    def __init__(self, dim: int = 256):
        super(ReidDescriptor50, self).__init__()
        self.resnet50_backbone = torch.nn.Sequential(
            *list(torchvision.models.resnet50().children())[:-1]
        )
        self.linear = torch.nn.Linear(2048, dim)
        self.softmax = torch.nn.Softmax()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.resnet50_backbone(x)
        y = torch.flatten(y, start_dim=1)
        y = self.linear(y)
        y = self.softmax(y)
        return y


