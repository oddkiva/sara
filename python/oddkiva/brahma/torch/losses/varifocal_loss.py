import torch


class VarifocalLoss(torch.nn.Module):
    """The Varifocal Loss implements the loss function described in
    [VarifocalNet: An IoU-aware Dense Object Detector](https://arxiv.org/pdf/2008.13367)
    """

    def __init__(self, alpha: float, gamma: float = 2.):
        """
        Parameters
        ----------
        alpha: scaling factor.
        gamma: dynamic weighting exponential factor.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        dyn_cls_weight = (1 - probs) ** self.gamma
        anti_dyn_cls_weight = probs ** self.gamma

        focal_loss = -self.alpha \
            * dyn_cls_weight * torch.log(probs)
        anti_focal_loss = (1 - self.alpha) \
            * anti_dyn_cls_weight * torch.log(1 - probs)
        positive_mask = (targets == 1)
        negative_mask = ~positive_mask

        return positive_mask * focal_loss + negative_mask * anti_focal_loss
