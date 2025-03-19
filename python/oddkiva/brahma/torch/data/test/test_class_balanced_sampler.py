from typing import Dict

from oddkiva.brahma.torch.data.class_balanced_sampler import ClassBalancedSampler


def test_class_balanced_sampler():
    # Generate a set of sample IDs grouped by class ID.
    sample_ids_grouped_by_class = [
        # Class 0
        list(range(1)),
        # Class 1
        list(range(1, 1 + 5)),
        # Class 2
        list(range(6, 6 + 20))
    ]
    
    # Flatten the set of sample IDs.
    sample_ids_flat = [
        id
        for group in sample_ids_grouped_by_class
        for id in group
    ]
    
    # lookup table:
    #   sample ID |-> class ID
    sample_labels: Dict[int, int] = {
        sample_id: class_idx
        for class_idx, group in enumerate(sample_ids_grouped_by_class)
        for sample_id in group
    }
    
    # The class-balanced sampler is a generator object.
    repeat = 100
    sample_generator = ClassBalancedSampler(sample_ids_grouped_by_class,
                                            repeat=repeat)

    # Generate the samples from the random sample generator
    for _ in range(20):
        samples = [*sample_generator]

        # Count the labels.
        labels = [sample_labels[x] for x in samples]
        label_histogram = [labels.count(i) for i in range(3)]
        print(f'\nlabel_histogram = {label_histogram}')

        # Each class must be sampled as uniformly as possible.
        #
        # A such criterion is the following score.
        # The closer the score is to 1, the better it is.
        class_sampling_uniformity_score = \
            min(label_histogram) / max(label_histogram)
        assert class_sampling_uniformity_score > 0.85
        print(f'uniformity sampling score = {class_sampling_uniformity_score}')
        
        id_histogram = [0] * len(sample_ids_flat)
        for sample_id in samples:
            id_histogram[sample_id] += 1
        assert all(count != 0 for count in id_histogram)
        
        a = min(enumerate(id_histogram), key=lambda v: v[1])
        b = max(enumerate(id_histogram), key=lambda v: v[1])
        print(f'id_histogram =\n{id_histogram}')
        print(f'least frequently sampled ID = {a[0]} count: {a[1]}')
        print(f'most frequently sampled ID = {b[0]} count: {b[1]}')
