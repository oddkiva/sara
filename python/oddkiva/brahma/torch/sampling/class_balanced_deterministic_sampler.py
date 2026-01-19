# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from typing import Iterator, List

from torch.utils.data import Sampler


class ClassBalancedDeterministicSampler(Sampler[int]):
    def __init__(
        self, sample_ids_grouped_by_class: List[List[int]]
    ):
        self.sample_ids_grouped_by_class = sample_ids_grouped_by_class
        self.class_count: int = len(sample_ids_grouped_by_class)
        self.sample_counts_per_class = [
            len(ixs) for ixs in sample_ids_grouped_by_class
        ]
        self.sample_count: int = sum(self.sample_counts_per_class)

        self.sample_count_max = max(self.sample_counts_per_class)
        self.class_most_populous = self.sample_counts_per_class.index(
            self.sample_count_max
        )

        self.sample_count_balanced = self.class_count * self.sample_count_max

    def __len__(self) -> int:
        return self.sample_count_balanced

    def __iter__(self) -> Iterator[int]:
        current_idx_for_class = [0] * self.class_count

        for i in range(self.sample_count_balanced):
            # The class to consider.
            class_idx = i % self.class_count

            # The current index for the current class.
            sample_idx = current_idx_for_class[class_idx]

            # Increment the iterator before yielding an iterator.
            current_idx_for_class[class_idx] = \
                (sample_idx + 1) % self.sample_counts_per_class[class_idx]

            sample_ids_for_class = self.sample_ids_grouped_by_class[class_idx]
            current_sample_id = sample_ids_for_class[sample_idx]
            yield current_sample_id
