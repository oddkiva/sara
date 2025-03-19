import torch
from torch.utils.data import Dataset, WeightedRandomSampler


class TripletDatabaseBase(Dataset):


    def __init__(self, base_dataset: Dataset, repeat: int = 1):
        self.base_dataset = base_dataset
        self.repeat = repeat

        self.partition_dataset_by_classes()
        self.generate_triplet_samples()

    def group_samples_by_class(self):
        self.class_count = len({class_id for _, class_id in self.base_dataset})

        self.samples_grouped_by_class = [[]] * self.class_count
        for sample_id, (_, class_id) in enumerate(self.base_dataset):
            self.samples_grouped_by_class[class_id].append(sample_id)

        self.sample_count_per_class = [
            len(group)
            for group in self.samples_grouped_by_class
        ]

    def generate_triplet_samples(self):
        self.weights = [1 / sample_count
                        for sample_count in self.sample_count_per_class]
        self.most_populous_class = max(enumerate(self.sample_count_per_class),
                                       key=lambda v: v[1])
        self.num_samples = \
            self.repeat * self.most_populous_class[1] * self.class_count
        self.sampler = WeightedRandomSampler(self.weights, self.num_samples)


        class_pairs = [] 
        for _ in range(self.num_samples):
            class_pairs.append(torch.randperm(self.class_count)[:2])

        triplets = []
        for class_pair in class_pairs:
            class_a, class_b = class_pair
            class_a_count = self.sample_count_per_class[class_a]
            class_b_count = self.sample_count_per_class[class_b]

            anchor, positive = torch.randperm(class_a_count)[:2]
            negative = torch.randint(0, class_b_count)

            anchor_id = self.samples_grouped_by_class[class_a][anchor]
            positive_id = self.samples_grouped_by_class[class_a][positive]
            negative_id = self.samples_grouped_by_class[class_b][negative]

            triplets.append([anchor_id, positive_id, negative_id])

