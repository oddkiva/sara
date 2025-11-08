import torch


class QuerySelector(torch.nn.Module):
    """
    This query selector selects queries based on the value of its most likely
    object class.
    """

    def __init__(self, K: int = 300):
        super().__init__()
        self.K = K

    @torch.no_grad()
    def forward(
        self,
        query: torch.Tensor,
        object_probs: torch.Tensor
    ) -> torch.Tensor:
        pass
