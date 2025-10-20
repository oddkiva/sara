from typing import List

from oddkiva.brahma.common.classification_dataset_abc import (
    ClassificationDatasetABC
)


def group_samples_by_class(
    dataset: ClassificationDatasetABC
) -> List[List[int]]:
    # Group samples by class.
    samples_grouped_by_class_dict = {}
    for sample_id, class_id in enumerate(dataset.image_class_ids):
        if class_id not in samples_grouped_by_class_dict:
            samples_grouped_by_class_dict[class_id] = [sample_id]
        else:
            samples_grouped_by_class_dict[class_id].append(sample_id)

    samples_grouped_by_class = []
    for class_id in range(dataset.class_count):
        samples_grouped_by_class.append(samples_grouped_by_class_dict[class_id])

    return samples_grouped_by_class


def check_class_statistics(sample_ids: List[int],
                           dataset: ClassificationDatasetABC):
    class_histogram = [0] * dataset.class_count
    for sample_id in sample_ids:
        class_id = dataset.image_class_ids[sample_id]
        class_histogram[class_id] += 1
    a = min(enumerate(class_histogram), key=lambda v: v[1])
    b = max(enumerate(class_histogram), key=lambda v: v[1])
    uniform_sampling_score = a[1] / b[1]

    print('class histogram=\n', class_histogram)
    print(f'least frequently sampled class ID: {a[0]}, count: {a[1]}')
    print(f'most  frequently sampled class ID: {b[0]}, count: {b[1]}')
    print(f'uniform_sampling_score = {uniform_sampling_score}')
