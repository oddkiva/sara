import torch


class FocalLoss(torch.nn.Module):
    """The focal loss.
    """

    def __init__(self, gamma: float = 2.):
        super().__init__()
        self.gamma = gamma

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """ 
        Parameters
        ----------
        outputs: outputs must be normalized in the range [0, 1]

        targets: zeros or ones.
        """
        # The closer to 1, the outputs is, the smaller dynamic weight become.
        dynamic_weight = (1 - outputs) ** self.gamma
        logits = torch.log(outputs)
        errors = dynamic_weight * logits
        return targets * errors
