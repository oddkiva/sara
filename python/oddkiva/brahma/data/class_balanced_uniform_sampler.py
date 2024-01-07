from typing import Iterator, List

import torch
from torch.utils.data import Sampler


class ClassBalancedSampler(Sampler[int]):
    def __init__(
        self, ids_partitioned_by_class: List[List[int]], num_samples: int
    ):
        self.ids_partitioned_by_class = ids_partitioned_by_class
        self.class_count: int = len(ids_partitioned_by_class)
        self.id_counts_per_class = torch.LongTensor(
            [len(ixs) for ixs in ids_partitioned_by_class]
        )
        self.id_count: int = torch.sum(self.id_counts_per_class).item()
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.id_count

    def __iter__(self) -> Iterator[int]:
        # Sample the classes uniformly.
        sampled_classes = torch.randint(
            0, self.class_count, (self.num_samples,)
        )
        # print(f'sampled classes = {sampled_classes}')

        # Fetch the cardinality of each sampled class.
        id_counts_in_sampled_classes = self.id_counts_per_class[
            sampled_classes
        ]
        # print(id_counts_in_sampled_classes)

        # Sample a data sample within each sampled class.
        # Draw a sample index within each class.
        sample_ixs = [
            torch.randint(0, id_count, (1,)).item()
            for id_count in id_counts_in_sampled_classes
        ]
        # print(sample_ixs)

        # From the sample index, get the actual sample "global" ID.
        sample_ids = [
            self.ids_partitioned_by_class[c][ix]
            for c, ix in zip(sampled_classes, sample_ixs)
        ]
        # print(sample_ids)

        for i in range(self.num_samples):
            yield sample_ids[i]


class ClassBalancedSampler2(Sampler[int]):
    """
    Simpler implementation with the same statistical properties.
    """

    def __init__(self, sample_labels: List[int], batch_size: int = 1):
        super().__init__()

        self.sample_count = len(sample_labels)
        self.label_count = max(sample_labels) + 1
        print(f"label_count = {self.label_count}")

        sample_ids = range(self.sample_count)
        labels = range(self.label_count)
        print(f"sample_ids = {sample_ids}")

        self.sample_ids_partitioned_by_label = [
            [
                sample_id
                for sample_id in sample_ids
                if sample_labels[sample_id] == l
            ]
            for l in labels
        ]
        print(
            f"sample_ids_partitioned_by_label = {self.sample_ids_partitioned_by_label}"
        )

        self.label_set_cardinalities = [
            len(self.sample_ids_partitioned_by_label[l]) for l in labels
        ]
        print(f"label_set_cardinalities = {self.label_set_cardinalities}")

        self.sample_weights = 1 / torch.Tensor(
            [self.label_set_cardinalities[l] for l in sample_labels]
        )
        print(f"sample_weights = {self.sample_weights}")

        self.sampler = torch.utils.data.sampler.WeightedRandomSampler(
            self.sample_weights, batch_size, replacement=False
        )

        def __len__(self) -> int:
            return len(self.sampler)

        def __iter__(self):
            yield from self.sampler


sample_labels = [0, 1, 1, 2, 2, 2]
sampler = ClassBalancedSampler2(sample_labels)

import IPython

IPython.embed()
