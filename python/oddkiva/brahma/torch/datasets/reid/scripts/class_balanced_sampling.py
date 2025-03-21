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

# Dataset
if platform.system() == 'Darwin':
    root_path = Path('/Users/oddkiva/Downloads/reid/dataset_ETHZ/')
else:
    root_path = Path('/home/david/GitLab/oddkiva/sara/data/reid/dataset_ETHZ/')
# Data transform
transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((160, 64), antialias=True)
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

batch_size = 32
train_dataset = ds
train_dataloader = DataLoader(train_dataset,
                              sampler=sample_gen,
                              batch_size=batch_size)

resnet50 = torchvision.models.resnet50(num_classes=ds.class_count)
loss_fn = torch.nn.CrossEntropyLoss()
writer = SummaryWriter('train/eth123')
write_interval = 1
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

        if step % write_interval == 0:
            # Train loss
            loss = loss.item()

            # Train classification error.
            label_pred = torch.argmax(pred, dim=1)
            class_error = torch.sum(label_pred != y) / batch_size

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
