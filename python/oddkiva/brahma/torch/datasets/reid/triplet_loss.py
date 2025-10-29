# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import logging
from rich.logging import RichHandler

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from oddkiva.brahma.torch.utils.logging import format_msg
from oddkiva.brahma.torch.parallel.ddp import get_local_rank


LOGGER = logging.getLogger(__name__)


class TripletLoss(torch.nn.Module):

    def __init__(self,
                 alpha: float = 0.5,
                 weight_decay_coeff: float = 1. / 5e4,
                 summary_writer: SummaryWriter | None = None,
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
                negative_desc: torch.Tensor) -> torch.Tensor:
        # negative_desc: torch.Tensor,
        # model_params: List[torch.nn.Parameter]) -> torch.Tensor:
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
        triplet_loss = torch.maximum(d_ap + (self.alpha - d_an), torch.tensor(0))
        mean_triplet_loss = torch.mean(triplet_loss)

        with torch.no_grad():
            # logd(LOGGER, 'triplet_loss', triplet_loss)
            LOGGER.debug(format_msg(
                f'min_d_ap = {torch.min(d_ap)}  max_d_ap = {torch.max(d_ap)}'))
            LOGGER.debug(format_msg(
                f'min_d_an = {torch.min(d_an)}  max_d_an = {torch.max(d_an)}'))
            LOGGER.debug(format_msg(f'mean_tl = {mean_triplet_loss}'))
            # logd(LOGGER, 'wts =', weight_decay)
            # logd(LOGGER, 'mtl_rw', mtl_rw)
            # logd(LOGGER, 'wts_rw =', regularization)

            # # Also small coefficients. So add this L2-norm regularized.
            # model_wts_norm = torch.tensor(0)
            # for v in model_params:
            #     model_wts_norm = model_wts_norm + 0.5 * torch.sum(v ** 2)
            # logd(LOGGER, 'model_wts', model_wts_norm)

        self.step = self.step + 1
        if get_local_rank() is None or get_local_rank() == 0:
            self._write_summaries(triplet_loss, d_ap, d_an)

        return mean_triplet_loss

    # regularization = self.weight_decay_coeff * weight_decay

    # weight_triplet_loss = 10
    # mtl_rw = weight_triplet_loss * mean_triplet_loss

    # regularized_triplet_loss = mtl_rw + regularization

    # if self.summary_writer is not None and \
    #         self.step % self.summary_write_interval == 0:
    # self._write_summaries(triplet_loss,
    #                       d_ap,
    #                       d_an,
    #                       weight_decay,
    #                       regularized_triplet_loss)
    # if self.train_with_regularization:
    #     return regularized_triplet_loss
    # else:
    #     return mean_triplet_loss

    def _write_summaries(self,
                         triplet_loss: torch.Tensor,
                         d_ap: torch.Tensor,
                         d_an: torch.Tensor):
        # weight_decay: torch.Tensor,
        # regularized_triplet_loss: torch.Tensor) -> None:

        if self.summary_writer is None:
            return

        with torch.no_grad():
            self.summary_writer.add_scalar('Train/mean_triplet_loss',
                                           torch.mean(triplet_loss), self.step)
            self.summary_writer.add_scalar('Train/mean_d_ap', torch.mean(d_ap), self.step)
            self.summary_writer.add_scalar('Train/mean_d_an', torch.mean(d_an), self.step)

            self.summary_writer.add_scalar('Train/min_triplet_loss',
                                           torch.min(triplet_loss), self.step)
            self.summary_writer.add_scalar('Train/max_triplet_loss',
                                           torch.max(triplet_loss), self.step)

            self.summary_writer.add_scalar('Train/max_d_ap', torch.max(d_ap), self.step)
            self.summary_writer.add_scalar('Train/min_d_an', torch.min(d_an), self.step)
            # self.summary_writer.add_scalar('TripletLoss/weight_decay', weight_decay, self.step)
            # self.summary_writer.add_scalar('TripletLoss/regularized_triplet_loss',
            #                                regularized_triplet_loss, self.step)
