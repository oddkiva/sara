from pathlib import Path

import torch
import torchvision
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from oddkiva.brahma.torch.datasets.reid.eth123 import ETH123

# Dataset
root_path = Path('/Users/oddkiva/Downloads/reid/dataset_ETHZ/')
transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((160, 24))
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

# NON-BATCHED balanced random sampling
repeat = 10
num_samples = ds.class_count * max(class_sample_counts) * repeat
sample_generator = WeightedRandomSampler(weights_for_sample, num_samples)
sample_ids = [*sample_generator]

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

train_dataset = ds
train_dataloader = DataLoader(train_dataset, sampler=sample_generator)

X, y = next(iter(train_dataloader))

resnet50 = torchvision.models.resnet50()
resnet50_backbone = list(resnet50.children())[:-1]
resnet50_bb = torch.nn.Sequential(*resnet50_backbone)
resnet50_reid = torch.nn.Sequential(*resnet50_backbone,
                                    torch.nn.Linear(2048, 1000))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet50_reid.parameters())

import IPython; IPython.embed()


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

        if sample_id % 100 == 0:
            loss, current = loss.item(), sample_id + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# train_loop(train_dataloader, resnet50_reid, loss_fn, optimizer)
