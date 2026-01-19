# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from typing import List

import torch
from torch.utils.data import Dataset

from oddkiva.brahma.torch.datasets.reid.triplet_dataset import TripletDataset


class MultiTripletDataset(Dataset):

    def __init__(self, triplet_datasets: List[TripletDataset],
                 num_dataset_samples: int):
        self.triplet_datasets = triplet_datasets

        num_datasets = len(self.triplet_datasets)
        self.num_dataset_samples = num_dataset_samples
        self.dataset_samples = torch.randint(0, num_datasets, (num_dataset_samples,))

        # Track the current indices for each triplet datasets.
        self._current_triplet_sample_idx = [0] * num_dataset_samples

    def  __len__(self):
        return self.num_dataset_samples * ([len(tds) for tds in self.triplet_datasets])

    def __getitem__(self, idx: int) -> TripletDataset.TripletSample:
        tds_idx = self.dataset_samples[idx]
        tds = self.triplet_datasets[tds_idx]

        # Get the triplet sample.
        triplet_sample_idx = self._current_triplet_sample_idx[tds_idx]
        triplet = tds[triplet_sample_idx]

        # Increment the current triplet sample index for the next queries.
        self._current_triplet_sample_idx[tds_idx] = \
            (triplet_sample_idx + 1) % len(tds)

        return triplet
