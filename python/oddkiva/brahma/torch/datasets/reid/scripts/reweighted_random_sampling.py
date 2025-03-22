# N.B.: I am not yet convinced with the WeightedRandomSampler class...
# TODO: check the code again.

import platform
from pathlib import Path

import torch
import torchvision
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter

from oddkiva.brahma.torch.datasets.reid.eth123 import ETH123

# Dataset
if platform.system() == 'Darwin':
    root_path = Path('/Users/oddkiva/Downloads/reid/dataset_ETHZ/')
else:
    root_path = Path('/home/david/GitLab/oddkiva/sara/data/reid/dataset_ETHZ/')
transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((160, 64))
])
ds = ETH123(root_path, transform=transform)

# Dataset statistics.
#
# Generate the class sampling weights.
class_sample_counts = [len(samples) for samples in ds._image_paths_per_class]
weights_for_class = [1 / count for count in class_sample_counts]
weights_for_sample = [
    weights_for_class[class_id]
    for class_id in ds._image_label_ixs
]
print(class_sample_counts)
print(weights_for_class)
print(len(weights_for_sample))

# NON-BATCHED balanced random sampling
repeat = 10
num_samples = ds.class_count * max(class_sample_counts) * repeat

class_generator = WeightedRandomSampler(weights_for_class, num_samples)

sample_generator = WeightedRandomSampler(weights_for_sample, num_samples)
sample_ids = [*sample_generator]

class_histogram = {}
for sample_id in sample_ids:
    class_id = ds.image_class_ids[sample_id]
    if class_id not in class_histogram:
        class_histogram[class_id] = 0
    else:
        class_histogram[class_id] += 1
class_histogram = [class_histogram[class_id]
                   for class_id in sorted(class_histogram.keys())]
a = min(enumerate(class_histogram), key=lambda v: v[1])
b = max(enumerate(class_histogram), key=lambda v: v[1])
uniform_sampling_score = a[1] / b[1]
print('class histogram=\n', class_histogram)
print(f'least frequently sampled class ID: {a[0]}, count: {a[1]}')
print(f'most  frequently sampled class ID: {b[0]}, count: {b[1]}')
print(f'uniform_sampling_score = {uniform_sampling_score}')

import IPython; IPython.embed()
exit()

train_dataset = ds
train_dataloader = DataLoader(
    train_dataset,
    sampler=WeightedRandomSampler(weights_for_sample, num_samples)
)

resnet50 = torchvision.models.resnet50(num_classes=ds.class_count)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet50.parameters())

writer = SummaryWriter('train/eth123')


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and
    # dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for sample_id, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if sample_id % 10 == 0:
            loss, current = loss.item(), sample_id + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            img = torchvision.utils.make_grid(X)
            writer.add_image('train image', img)


# train_loop(train_dataloader, resnet50, loss_fn, optimizer)
