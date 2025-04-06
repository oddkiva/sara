from typing import List, Optional

import torch
from torch.utils.tensorboard.writer import SummaryWriter


class TripletLoss(torch.nn.Module):

    def __init__(self,
                 alpha: float = 2e-1,
                 weight_decay_coeff: float = 5e-4,
                 summary_writer: Optional[SummaryWriter] = None,
                 summary_write_interval: int = 10):
        self.alpha = alpha
        self.weight_decay_coeff = weight_decay_coeff

        self.summary_writer = summary_writer
        self.step = 0
        self.summary_write_interval = summary_write_interval

    def forward(self,
                anchor_desc: torch.Tensor,
                positive_desc: torch.Tensor,
                negative_desc: torch.Tensor,
                model_params: List[torch.nn.Parameter]) -> torch.Tensor:
        d_ap = torch.sum((anchor_desc - positive_desc) ** 2, dim=-1)
        d_an = torch.sum((anchor_desc - negative_desc) ** 2, dim=-1)

        # The triplet loss can be zero.
        triplet_loss = torch.maximum(self.alpha + d_ap - d_an, torch.tensor(0))
        mean_triplet_loss = torch.mean(triplet_loss)

        # Also small coefficients. So add this L2-norm regularized.
        weight_decay = torch.tensor(0)
        for v in model_params:
            weight_decay += 0.5 * torch.sum(v ** 2)
        regularized_triplet_loss = mean_triplet_loss + \
            self.weight_decay_coeff * weight_decay

        self.step += 1
        if self.summary_writer is not None and \
                self.step % self.summary_write_interval == 0:
            self._write_summaries(triplet_loss, d_ap, d_an, weight_decay,
                                  regularized_triplet_loss)

        return regularized_triplet_loss


    def _write_summaries(self,
                         triplet_loss: torch.Tensor,
                         d_ap: torch.Tensor,
                         d_an: torch.Tensor,
                         weight_decay: torch.Tensor,
                         regularized_triplet_loss: torch.Tensor):
        if self.summary_writer is None:
            return

        self.summary_writer.add_scalar('triplet_loss', triplet_loss, self.step)
        self.summary_writer.add_scalar('d_ap', torch.mean(d_ap), self.step)
        self.summary_writer.add_scalar('d_an', torch.mean(d_an), self.step)
        self.summary_writer.add_scalar('weight_decay', weight_decay, self.step)
        self.summary_writer.add_scalar('regularized_triplet_loss',
                                       regularized_triplet_loss, self.step)
