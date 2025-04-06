from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from oddkiva.brahma.torch.datasets.classification_dataset_abc import (
    ClassificationDatasetABC
)

class TripletDatabase(Dataset):

    TripletSample = Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                          Tuple[int, int, int]]

    def __init__(self, base_dataset: ClassificationDatasetABC, repeat: int = 1):
        self.base_dataset = base_dataset
        self.repeat = repeat

        self.group_samples_by_class()
        self.generate_triplet_samples()

    def group_samples_by_class(self):
        # Group samples by class.
        samples_grouped_by_class_dict = {}
        for sample_id, class_id in enumerate(self.base_dataset.image_class_ids):
            if class_id not in samples_grouped_by_class_dict:
                samples_grouped_by_class_dict[class_id] = [sample_id]
            else:
                samples_grouped_by_class_dict[class_id].append(sample_id)

        samples_grouped_by_class = []
        for class_id in range(self.base_dataset.class_count):
            samples_grouped_by_class.append(samples_grouped_by_class_dict[class_id])
        self.samples_grouped_by_class = samples_grouped_by_class

        # Class statistics.
        self.sample_counts_per_class = torch.LongTensor(
            [len(ixs) for ixs in samples_grouped_by_class]
        )
        self.sample_count: int = int(
            torch.sum(self.sample_counts_per_class).item()
        )
        self.sample_count_max: int = int(
            torch.max(self.sample_counts_per_class).item()
        )
        self.sample_count_balanced = \
            self.base_dataset.class_count * self.sample_count_max * self.repeat

    def generate_triplet_samples(self):
        # Draw two distincts class indices.
        class_pairs = []
        for _ in range(self.sample_count_balanced):
            class_pairs.append(
                torch.randperm(self.base_dataset.class_count)[:2]
            )

        # Draw triplets of sample indices.
        triplets = []
        for class_pair in class_pairs:
            class_a, class_b = [int(v.item()) for v in class_pair]
            class_a_count = int(self.sample_counts_per_class[class_a].item())
            class_b_count = int(self.sample_counts_per_class[class_b].item())

            # Anchor and positive indices.
            ap_tensor = torch.randperm(class_a_count)[:2]
            anchor, positive = [int(v.item()) for v in ap_tensor]

            # Negative index.
            n_tensor = torch.randint(0, class_b_count, (1,))
            negative = int(n_tensor[0].item())

            anchor_gid = self.samples_grouped_by_class[class_a][anchor]
            positive_gid = self.samples_grouped_by_class[class_a][positive]
            negative_gid = self.samples_grouped_by_class[class_b][negative]

            triplets.append((anchor_gid, positive_gid, negative_gid))

        self.triplets = triplets


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
