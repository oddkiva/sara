import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    """The focal loss.
    """

    def __init__(self, gamma: float = 2, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        r""" 
        Parameters:
            scores: score values be normalized in the range $[0, 1]$.
            targets: zeros or ones.
        """
        # The dynamic weights that gives less importance to easy-to-classify
        # training samples.
        p_t = targets * scores + (1 - targets) * (1 - scores)

        ce_loss = F.binary_cross_entropy_with_logits(scores, targets, reduce=None)
        # The dynamically re-weighted cross-entropy loss
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss

        # Dynamically re-balance the focal loss between the negative and
        # positive samples.
        if 0 <= self.alpha and self.alpha <= 1:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_loss = alpha_t * focal_loss

        return focal_loss
