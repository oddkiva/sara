import torch
import torch.nn as nn


class AnchorSelector(nn.Module):
    r"""
    Selects the top-$K$ object queries and also returns with their associated class logits and
    geometry logits.
    """

    def __init__(self, top_K: int = 300):
        super().__init__()
        self._top_K = top_K

    def forward(
        self,
        memory: torch.Tensor,
        class_logits: torch.Tensor,
        geometry_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_class_logits = class_logits.max(dim=-1).values

        _, topk_ind = torch.topk(
            max_class_logits,
            self._top_K,
            dim=-1
        )

        class_dim = class_logits.shape[-1]
        geometry_dim = geometry_logits.shape[-1]
        embedding_dim = memory.shape[-1]

        ixs = topk_ind.unsqueeze(-1).repeat(1, 1, geometry_dim)
        topk_coords = geometry_logits.gather(dim=1, index=ixs)

        ixs = topk_ind.unsqueeze(-1).repeat(1, 1, class_dim)
        topk_logits = class_logits.gather(dim=1, index=ixs)

        ixs = topk_ind.unsqueeze(-1).repeat(1, 1, embedding_dim)
        topk_memory = memory.gather(dim=1, index=ixs)

        return topk_memory, topk_logits, topk_coords
