import torch
import torch.nn as nn


class AnchorSelector(nn.Module):

    def __init__(self, top_k: int = 300):
        super().__init__()
        self._top_k = top_k

    def forward(
        self,
        memory: torch.Tensor,
        query_class_logits: torch.Tensor,
        query_geometries_unactivated: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, topk_ind = torch.topk(
            query_class_logits.max(-1).values,
            self._top_k,
            dim=-1
        )

        ixs = topk_ind\
            .unsqueeze(-1)\
            .repeat(1, 1, query_geometries_unactivated.shape[-1])
        topk_coords = query_geometries_unactivated.gather(dim=1, index=ixs)

        ixs = topk_ind.unsqueeze(-1).repeat(1, 1, query_class_logits.shape[-1])
        topk_logits = query_class_logits.gather( dim=1, index=ixs)

        ixs = topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1])
        topk_memory = memory.gather( dim=1, index=ixs)

        return topk_memory, topk_logits, topk_coords
