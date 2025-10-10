from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.data.class_balanced_deterministic_sampler import (
    ClassBalancedDeterministicSampler
)
from oddkiva.brahma.torch.datasets.reid.eth123 import ETH123
from oddkiva.brahma.torch.datasets.utils import (
    check_class_statistics,
    group_samples_by_class
)

root_path = DATA_DIR_PATH / 'reid' / 'dataset_ETHZ'
ds = ETH123(root_path)

samples_grouped_by_class = group_samples_by_class(ds)
sample_gen = ClassBalancedDeterministicSampler(samples_grouped_by_class)
sample_ids = [*sample_gen]

check_class_statistics(sample_ids, ds)
