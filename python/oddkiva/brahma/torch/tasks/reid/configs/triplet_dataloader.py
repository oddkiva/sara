# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from oddkiva.brahma.torch.parallel.ddp import torchrun_is_running
from oddkiva.brahma.torch.datasets.reid.triplet_dataset import (
    TripletDataset
)


def make_dataloader_for_triplet_loss(
    tds: TripletDataset,
    batch_size: int
) -> DataLoader:
    if torchrun_is_running():
        return DataLoader(
            dataset=tds,
            batch_size=batch_size,
            # The following options are for parallel data training
            shuffle=False,
            sampler=DistributedSampler(tds)
        )
    else:
        return DataLoader(
            dataset=tds,
            batch_size=batch_size,
            # The triplet dataset already samples randomly, so no point.
            shuffle=False
        )
