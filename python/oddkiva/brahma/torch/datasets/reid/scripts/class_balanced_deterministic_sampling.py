import platform
from pathlib import Path

from oddkiva.brahma.torch.datasets.reid.eth123 import ETH123
from oddkiva.brahma.torch.data.class_balanced_deterministic_sampler import (
    ClassBalancedDeterministicSampler)

# Dataset
if platform.system() == 'Darwin':
    root_path = Path('/Users/oddkiva/Downloads/reid/dataset_ETHZ/')
else:
    root_path = Path('/home/david/GitLab/oddkiva/sara/data/reid/dataset_ETHZ/')
ds = ETH123(root_path)

samples_grouped_by_class_dict = {}
for sample_id, class_id in enumerate(ds.image_class_ids):
    if class_id not in samples_grouped_by_class_dict:
        samples_grouped_by_class_dict[class_id] = [sample_id]
    else:
        samples_grouped_by_class_dict[class_id].append(sample_id)

samples_grouped_by_class = []
for class_id in range(ds.class_count):
    samples_grouped_by_class.append(samples_grouped_by_class_dict[class_id])
    
sample_gen = ClassBalancedDeterministicSampler(samples_grouped_by_class)
sample_ids = [*sample_gen]

class_histogram = [0] * ds.class_count
for sample_id in sample_ids:
    class_id = ds.image_class_ids[sample_id]
    class_histogram[class_id] += 1
a = min(enumerate(class_histogram), key=lambda v: v[1])
b = max(enumerate(class_histogram), key=lambda v: v[1])
uniform_sampling_score = a[1] / b[1]
print(f'least frequently sampled class ID: {a[0]}, count: {a[1]}')
print(f'most  frequently sampled class ID: {b[0]}, count: {b[1]}')
print(f'uniform_sampling_score = {uniform_sampling_score}')
