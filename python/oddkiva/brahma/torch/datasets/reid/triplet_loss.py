from typing import List, Optional

import torch
from torch.utils.tensorboard.writer import SummaryWriter


class TripletLoss(torch.nn.Module):

    def __init__(self,
                 alpha: float = 5e-1,
                 weight_decay_coeff: float = 1. / 5e4,
                 summary_writer: Optional[SummaryWriter] = None,
                 summary_write_interval: int = 10,
                 train_with_regularization: bool = False):
        super(TripletLoss, self).__init__()
        self.alpha = alpha
        self.weight_decay_coeff = weight_decay_coeff
        self.train_with_regularization = train_with_regularization

        self.summary_writer = summary_writer
        self.step = 0
        self.summary_write_interval = summary_write_interval

    def forward(self,
                anchor_desc: torch.Tensor,
                positive_desc: torch.Tensor,
                negative_desc: torch.Tensor,
                model_params: List[torch.nn.Parameter]) -> torch.Tensor:
        # Should be close to zero.
        d_ap = torch.sum((anchor_desc - positive_desc) ** 2, dim=-1)
        # Cannot be zero and must be a very large positive
        d_an = torch.sum((anchor_desc - negative_desc) ** 2, dim=-1)

        # The triplet loss.
        # When the train loss converges, we expect:
        #   d_ap == 0
        #   d_an >> 1
        #   triplet_loss <= self.alpha
        #
        # - A negative triplet loss provides good confidence in the training.
        # - A positive triplet loss means we need to wait a bit or tune a bit the
        #   coefficients (alpha, etc).
        triplet_loss = torch.maximum(self.alpha + d_ap - d_an, torch.tensor(0))
        mean_triplet_loss = torch.mean(triplet_loss)

        # Also small coefficients. So add this L2-norm regularized.
        weight_decay = torch.tensor(0)
        for v in model_params:
            weight_decay = weight_decay + 0.5 * torch.sum(v ** 2)

        regularization = self.weight_decay_coeff * weight_decay

        weight_triplet_loss = 10
        mtl_rw = weight_triplet_loss * mean_triplet_loss

        regularized_triplet_loss = mtl_rw + regularization

        # print('mtl =', mean_triplet_loss)
        # print('wts =', weight_decay)
        # print('mtl_rw', mtl_rw)
        # print('wts_rw =', regularization)

        self.step += 1
        if self.summary_writer is not None and \
                self.step % self.summary_write_interval == 0:
            self._write_summaries(torch.mean(triplet_loss),
                                  torch.mean(d_ap),
                                  torch.mean(d_an),
                                  weight_decay,
                                  regularized_triplet_loss)

        if self.train_with_regularization:
            return regularized_triplet_loss
        else:
            return mean_triplet_loss

    def _write_summaries(self,
                         triplet_loss: torch.Tensor,
                         d_ap: torch.Tensor,
                         d_an: torch.Tensor,
                         weight_decay: torch.Tensor,
                         regularized_triplet_loss: torch.Tensor):
        if self.summary_writer is None:
            return

        self.summary_writer.add_scalar('TripletLoss/triplet_loss', triplet_loss, self.step)
        self.summary_writer.add_scalar('TripletLoss/d_ap', torch.mean(d_ap), self.step)
        self.summary_writer.add_scalar('TripletLoss/d_an', torch.mean(d_an), self.step)
        self.summary_writer.add_scalar('TripletLoss/weight_decay', weight_decay, self.step)
        self.summary_writer.add_scalar('TripletLoss/regularized_triplet_loss',
                                       regularized_triplet_loss, self.step)
