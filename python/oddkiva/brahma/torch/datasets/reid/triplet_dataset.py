# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from loguru import logger

import torch
from torch.utils.data import Dataset

from oddkiva.brahma.common.classification_dataset_abc import (
    ClassificationDatasetABC
)
from oddkiva.brahma.torch.datasets.utils import group_samples_by_class
from oddkiva.brahma.torch.utils.logging import format_msg


class TripletDataset(Dataset):
    """The base triplet dataset class.
    """

    TripletSample = tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                          tuple[int, int, int]]

    def __init__(self, base_dataset: ClassificationDatasetABC, repeat: int = 1):
        r"""
        The `TripletDataset` class must be initialized from a class derived
        from `ClassificationDatasetABC`.

        Once instantiated, the `TripletDataset` object will generate triplet of
        samples randomly, where samples are drawn in such a way that each image
        class is sampled in an equidistributed manner, regardless of the
        cardinality of each image class.

        The number of triplets we sample is calculated as follows.

        First, let $N$ denote the number of images, $L$ denote the number of
        image classes, $l_n$ denote the label of image $n$. Then, we count the
        cardinality of the most frequent image class $l^*$.

        $$
        l^* = \underset{l}{\mathrm{argmax}}
            \ \mathrm{card} \left\{ n | l_n = l \right\}
        $$

        We introduce the *repeat* parameter $r$ so that we sample each class
        this number of times

        $$
        r \times \mathrm{card} \{ n | l_n = l^* \}
        $$

        Thus, the total number of triplet samples we draw is

        $$
        r  \times L \times \mathrm{card} \{ n | l_n = l^* \}
        $$

        Parameters:
            base_dataset:
                the classification dataset from which we sample triplets.
            repeat:
                the integer parameter $r$ in the formula.
        """
        self.base_dataset = base_dataset
        self.repeat = repeat

        logger.info(format_msg('Grouping samples by classes...'))
        self._group_samples_by_class()
        logger.info(format_msg('Generating triplet samples...'))
        self._generate_triplet_samples()

    def  __len__(self) -> int:
        """Returns the number of triplet samples
        """
        return len(self.triplets)

    def __getitem__(self, idx: int) -> TripletSample:
        r"""Returns the triplet samples indexed by the index `idx`.

        Parameters:
            idx: the triplet index

        - A sample is the image-label pair which we denote by
          $(\mathbf{I}_i, l_i)$.
        - A triplet is defined by 3 samples, namely, the anchor, positive
          and negative samples.
          Let us respectively index the anchor, positive and negative samples
          by the triplet of indices $(a, p, n)$.
          Then the anchor, positive and negative samples are respectively the
          pairs:

          - $(\mathbf{I}_a, l_a)$,
          - $(\mathbf{I}_p, l_p)$, and
          - $(\mathbf{I}_n, l_n)$.
        """
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
        logger.info(format_msg('Positive-negative class sampling...'))
        # We use the multinomial distribution instead of randperm as we can
        # leverage parallelization.
        #
        # positive_negative_class_pairs = []
        # for _ in range(self.sample_count_balanced):
        #     positive_negative_class_pairs.append(
        #         torch.randperm(self.base_dataset.class_count)[:2]
        #     )
        class_weights = torch.tensor(
            [1.0] * self.base_dataset.class_count
        ).expand(
            self.sample_count_balanced, -1
        )
        positive_negative_class_pairs = torch.multinomial(
            class_weights, num_samples=2, replacement=False)

        # Draw triplets of sample indices.
        logger.info(format_msg('Triplet sampling...'))
        triplets = []
        for class_pair in positive_negative_class_pairs:
            p_class, n_class = [int(v.item()) for v in class_pair]
            p_class_count = int(self.sample_counts_per_class[p_class].item())
            n_class_count = int(self.sample_counts_per_class[n_class].item())

            # This does happen and we simply ignore this case.
            if p_class_count < 2:
                continue

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
