from typing import Iterator, List

import torch
from torch.utils.data import Sampler


class ClassBalancedSampler(Sampler[int]):
    def __init__(
        self, sample_ids_grouped_by_class: List[List[int]],
        repeat: int = 1
    ):
        self.sample_ids_grouped_by_class = sample_ids_grouped_by_class
        self.class_count: int = len(sample_ids_grouped_by_class)
        self.sample_counts_per_class = torch.LongTensor(
            [len(ixs) for ixs in sample_ids_grouped_by_class]
        )
        self.sample_count: int = int(
            torch.sum(self.sample_counts_per_class).item()
        )
        self.sample_count_max: int = int(
            torch.max(self.sample_counts_per_class).item()
        )

        self.sample_count_balanced = \
            self.class_count * self.sample_count_max * repeat

    def __len__(self) -> int:
        return self.sample_count_balanced

    def __iter__(self) -> Iterator[int]:
        # Sample the class indices uniformly.
        sampled_classes = torch.randint(
            0, self.class_count, (self.sample_count_balanced,)
        )
        # print(f'sampled classes = {sampled_classes}')

        # Fetch the cardinality of each sampled class.
        sample_counts_in_sampled_classes = self.sample_counts_per_class[
            sampled_classes
        ]
        # print(
        #     f'sample_counts_in_sampled_classes = {sample_counts_in_sampled_classes}'
        # )

        # Sample a data sample within each sampled class.
        # Draw a sample index within each class.
        sample_ixs = [
            torch.randint(0, sample_count, (1,)).item()
            for sample_count in sample_counts_in_sampled_classes
        ]
        # print(f'sample_ixs = {sample_ixs}')

        # From the sample index, get the actual sample "global" ID.
        sample_ids = [
            self.sample_ids_grouped_by_class[c][ix]
            for c, ix in zip(sampled_classes, sample_ixs)
        ]
        # print(f'sample_ids = {sample_ids}')

        for i in range(self.sample_count_balanced):
            yield sample_ids[i]
