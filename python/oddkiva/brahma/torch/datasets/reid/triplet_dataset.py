from typing import Tuple

import torch
from torch.utils.data import Dataset

from oddkiva.brahma.torch.datasets.classification_dataset_abc import (
    ClassificationDatasetABC
)
from oddkiva.brahma.torch.datasets.utils import group_samples_by_class


class TripletDatabase(Dataset):

    TripletSample = Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                          Tuple[int, int, int]]

    def __init__(self, base_dataset: ClassificationDatasetABC, repeat: int = 1):
        self.base_dataset = base_dataset
        self.repeat = repeat

        self._group_samples_by_class()
        self._generate_triplet_samples()

    def  __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx: int) -> TripletSample:
        # Anchor, positive, negative samples.
        a, p, n = self.triplets[idx]

        # Unpack the image data and the image label.
        Xa, ya = self.base_dataset[a]
        Xp, yp = self.base_dataset[p]
        Xn, yn = self.base_dataset[n]

        return (Xa, Xp, Xn), (ya, yp, yn)

    def _group_samples_by_class(self):
        # Group samples by class.
        self.samples_grouped_by_class = group_samples_by_class(self.base_dataset)

        # Class statistics.
        self.sample_counts_per_class = torch.LongTensor(
            [len(ixs) for ixs in self.samples_grouped_by_class]
        )
        self.sample_count: int = int(
            torch.sum(self.sample_counts_per_class).item()
        )
        self.sample_count_max: int = int(
            torch.max(self.sample_counts_per_class).item()
        )
        self.sample_count_balanced = \
            self.base_dataset.class_count * self.sample_count_max * self.repeat

    def _generate_triplet_samples(self):
        # Draw two distincts class indices.
        positive_negative_class_pairs = []
        for _ in range(self.sample_count_balanced):
            positive_negative_class_pairs.append(
                torch.randperm(self.base_dataset.class_count)[:2]
            )

        # Draw triplets of sample indices.
        triplets = []
        for class_pair in positive_negative_class_pairs:
            p_class, n_class = [int(v.item()) for v in class_pair]
            p_class_count = int(self.sample_counts_per_class[p_class].item())
            n_class_count = int(self.sample_counts_per_class[n_class].item())

            # Draw a anchor and positive **local** indices (lid) as the indexing is
            # relative to the class group.
            ap_lids = torch.randperm(p_class_count)[:2]
            anchor, positive = [int(v.item()) for v in ap_lids]

            # Draw a negative local index.
            n_lid = torch.randint(0, n_class_count, (1,))
            negative = int(n_lid.item())

            # Extract the corresponding global indices.
            anchor_gid = self.samples_grouped_by_class[p_class][anchor]
            positive_gid = self.samples_grouped_by_class[p_class][positive]
            negative_gid = self.samples_grouped_by_class[n_class][negative]

            triplets.append((anchor_gid, positive_gid, negative_gid))

        self.triplets = triplets
