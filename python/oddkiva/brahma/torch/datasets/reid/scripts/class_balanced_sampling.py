import platform
from pathlib import Path
from typing import List

import torch
import torchvision
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from oddkiva.brahma.torch.datasets.reid.eth123 import ETH123
from oddkiva.brahma.torch.data.class_balanced_sampler import ClassBalancedSampler


class ModelConfig:
    image_size = (160, 64)
    batch_size = 32

    summary_write_interval = 1


# Dataset
if platform.system() == 'Darwin':
    root_path = Path('/Users/oddkiva/Downloads/reid/dataset_ETHZ/')
else:
    root_path = Path('/home/david/GitLab/oddkiva/sara/data/reid/dataset_ETHZ/')
# Data transform
transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(ModelConfig.image_size, antialias=True)
])
ds = ETH123(root_path, transform=transform)

# Group samples by class.
samples_grouped_by_class_dict = {}
for sample_id, class_id in enumerate(ds.image_class_ids):
    if class_id not in samples_grouped_by_class_dict:
        samples_grouped_by_class_dict[class_id] = [sample_id]
    else:
        samples_grouped_by_class_dict[class_id].append(sample_id)

samples_grouped_by_class = []
for class_id in range(ds.class_count):
    samples_grouped_by_class.append(samples_grouped_by_class_dict[class_id])

sample_gen = ClassBalancedSampler(samples_grouped_by_class, 1)
sample_ids = [*sample_gen]

def check_class_statistics(sample_ids: List[int]):
    class_histogram = [0] * ds.class_count
    for sample_id in sample_ids:
        class_id = ds.image_class_ids[sample_id]
        class_histogram[class_id] += 1
    a = min(enumerate(class_histogram), key=lambda v: v[1])
    b = max(enumerate(class_histogram), key=lambda v: v[1])
    uniform_sampling_score = a[1] / b[1]
    print('class histogram=\n', class_histogram)
    print(f'least frequently sampled class ID: {a[0]}, count: {a[1]}')
    print(f'most  frequently sampled class ID: {b[0]}, count: {b[1]}')
    print(f'uniform_sampling_score = {uniform_sampling_score}')
check_class_statistics(sample_ids)


class ReidDescriptor50(torch.nn.Module):

    def __init__(self, dim: int = 256):
        self.resnet50_backbone = torch.nn.Sequential(
            *list(torchvision.models.resnet50().children())[:-2]
        )
        self.linear = torch.nn.Linear(2048, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.resnet50_backbone(x)
        y = torch.flatten(y, start_dim=2)
        y = self.linear(y)
        return y


class HypersphericManifoldLoss(torch.nn.Module):

    def __init__(self, num_classes: int, batch_size: int, embedding_dim: int = 256):
        self.means = torch.empty(num_classes, embedding_dim)
        self.batch_size = batch_size
        torch.nn.init.kaiming_uniform_(self.means)

    def forward(self, input, target):
        input_normalized = input / torch.norm(input, -1)
        means = self.means[target]
        means_normalized = means / torch.norm(means, -1)

        # Maximize the cosine similarity between the input and the mean.
        cosines = torch.sum(input_normalized - means_normalized, -1)
        # Therefore minimize this positive number
        diff = (1 - cosines) * 0.5 / self.batch_size # The appropriate scaling.

        # Minimize the mutual cosine similarity between the means.
        target_unique = torch.unique(target)
        means_unique = self.means[target_unique]
        means_unique_normalized = \
            means_unique[target_unique] / torch.norm(means_unique, -1)
        mutual_mean_cosines = torch.matmul(means_unique_normalized,
                                           torch.t(means_unique_normalized))

        # There are at most (N-1) (N - 2)/ 2 cosine
        mutual_mean_cosines = \
            torch.triu(mutual_mean_cosines, diagonal=1) / self.batch_size

        # We don't want the coefficients of the means to explode towards
        # infinity either.
        regularization_l2 = torch.sum(self.means ** 2) / self.batch_size

        return diff + mutual_mean_cosines + regularization_l2


train_dataset = ds
train_dataloader = DataLoader(train_dataset,
                              sampler=sample_gen,
                              batch_size=ModelConfig.batch_size)

resnet50 = torchvision.models.resnet50(num_classes=ds.class_count)
loss_fn = torch.nn.CrossEntropyLoss()

writer = SummaryWriter('train/eth123')
class_histogram_1 = [0] * ds.class_count


def train_loop(dataloader, model, loss_fn, optimizer):
    step_count = len(dataloader)

    # Set the model to training mode - important for batch normalization and
    # dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for step, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update class statistics
        class_ids = y
        for class_id in class_ids:
            class_histogram_1[class_id] += 1

        if step % ModelConfig.summary_write_interval == 0:
            # Train loss
            loss = loss.item()

            # Train classification error.
            label_pred = torch.argmax(pred, dim=1)
            class_error = torch.sum(label_pred != y) / ModelConfig.batch_size

            img = torchvision.utils.make_grid(X)
            writer.add_image('Train/image', img, step)
            writer.add_scalar('Train/loss', loss, step)

            # Monitor the balanced random sampling
            a = min(enumerate(class_histogram_1), key=lambda v: v[1])
            b = max(enumerate(class_histogram_1), key=lambda v: v[1])
            uniform_sampling_score = a[1] / b[1]

            # Write for tensorboard.
            writer.add_scalar('ClassBalancedStats/least_frequent_class',
                              a[0], step)
            writer.add_scalar('ClassBalancedStats/least_frequent_class_count',
                              a[1], step)
            writer.add_scalar('ClassBalancedStats/most_frequent_class',
                              b[0], step)
            writer.add_scalar('ClassBalancedStats/most_frequent_class count',
                              b[1], step)
            writer.add_scalar('ClassBalancedStats/uniform_sampling_score',
                              uniform_sampling_score, step)
            writer.add_scalar('Train/classification_error', class_error, step)

            # Log on the console.
            print("".join([
                f"[iter: {step:>5d}/{step_count:>5d}] ",
                f"loss: {loss:>7f}, ",
                f"classification_error: {class_error:>7f} "
            ]))


for epoch in range(10):
    optimizer = torch.optim.Adam(resnet50.parameters())
    train_loop(train_dataloader, resnet50, loss_fn, optimizer)
    torch.save(resnet50.state_dict(), f'eth123_resnet50_{epoch}.pt')
