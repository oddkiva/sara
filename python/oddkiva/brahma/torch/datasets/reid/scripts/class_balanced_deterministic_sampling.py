import platform
from pathlib import Path

from oddkiva.brahma.torch.data.class_balanced_deterministic_sampler import (
    ClassBalancedDeterministicSampler
)
from oddkiva.brahma.torch.datasets.reid.eth123 import ETH123
from oddkiva.brahma.torch.datasets.utils import (
    check_class_statistics,
    group_samples_by_class
)

# Dataset
if platform.system() == 'Darwin':
    root_path = Path('/Users/oddkiva/Downloads/reid/dataset_ETHZ/')
else:
    root_path = Path('/home/david/GitLab/oddkiva/sara/data/reid/dataset_ETHZ/')
ds = ETH123(root_path)

samples_grouped_by_class = group_samples_by_class(ds)
sample_gen = ClassBalancedDeterministicSampler(samples_grouped_by_class)
sample_ids = [*sample_gen]

check_class_statistics(sample_ids, ds)
