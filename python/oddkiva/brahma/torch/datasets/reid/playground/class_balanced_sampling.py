# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torch
import torch.nn
import torchvision
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.sampling.class_balanced_sampler import (
    make_class_balanced_sampler
)
from oddkiva.brahma.torch.datasets.reid.eth123 import ETH123


class ModelConfig:
    reid_dim = 256
    image_size = (160, 64)
    batch_size = 32
    summary_write_interval = 1

    # Add distributed training configs
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # rank = int(os.environ.get("RANK", 0))


# Dataset
ROOT_PATH = DATA_DIR_PATH / 'reid' / 'dataset_ETHZ'
# Data transform
DATA_TRANSFORM = v2.Compose([
    v2.Resize(ModelConfig.image_size, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
])
ETH123_DS = ETH123(ROOT_PATH, transform=DATA_TRANSFORM)


class ReidDescriptor50(torch.nn.Module):

    def __init__(self, dim: int = 256):
        super(ReidDescriptor50, self).__init__()
        self.resnet50_backbone = torch.nn.Sequential(
            *list(torchvision.models.resnet50().children())[:-1]
        )
        self.linear = torch.nn.Linear(2048, dim)
        self.softmax = torch.nn.Softmax()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.resnet50_backbone(x)
        y = torch.flatten(y, start_dim=1)
        y = self.linear(y)
        y = self.softmax(y)
        return y


class HypersphericManifoldLoss(torch.nn.Module):

    def __init__(self, num_classes: int, batch_size: int, embedding_dim: int = 256):
        super(HypersphericManifoldLoss, self).__init__()
        with torch.no_grad():
            means = torch.empty(num_classes, embedding_dim)
            torch.nn.init.kaiming_uniform_(means)
            self.means = torch.nn.Parameter(means)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        input_n = input / torch.norm(input, dim=-1)[..., torch.newaxis]

        means = self.means
        means_n = means / torch.norm(means, dim=-1)[..., torch.newaxis]

        # Maximize cosine similarities
        cosines = torch.matmul(input_n, torch.t(means_n))
        # Think of the cosine similarity value as some sort of probability value.
        logit = torch.exp((1 + cosines) * 0.5)
        logit = logit / torch.sum(logit, 1)[..., torch.newaxis]

        target_vecs = torch.nn.functional.one_hot(target, self.num_classes)

        xentropy_loss = torch.sum(-logit * target_vecs) / self.batch_size

        # Also minimize mean inter-similarities
        antitarget_vecs = torch.ones(self.means.shape) - target_vecs
        means_intersims = torch.matmul(means_n, torch.t(means_n)) ** 2
        xentropy_loss2 = torch.sum(-torch.log(means_intersims) *
                                   antitarget_vecs) / (self.batch_size ** 2)

        # We don't want the coefficients of the means to explode towards
        # infinity either.
        regularization_l2 = torch.sum(self.means ** 2) / self.batch_size

        return xentropy_loss + xentropy_loss2 + regularization_l2


class NearestAssignmentLoss(torch.nn.Module):

    def __init__(self, means):
        super(NearestAssignmentLoss, self).__init__()
        self.means = means

    def forward(self, input, target):
        means = self.means[target]
        means_n = means / torch.norm(means, dim=-1)[..., torch.newaxis]
        input_n = input / torch.norm(input, dim=-1)[..., torch.newaxis]
        similarities = torch.matmul(input_n, torch.t(means_n))
        labels_assigned = torch.argmax(similarities, -1)
        labels_diff = torch.sum(labels_assigned != target)
        return labels_diff


# def train_loop(dataloader, model, loss_fn, optimizer):
#     step_count = len(dataloader)
#
#     # Set the model to training mode - important for batch normalization and
#     # dropout layers
#     # Unnecessary in this situation but added for best practices
#     model.train()
#     for step, (X, y) in enumerate(dataloader):
#         # Compute prediction and loss
#         pred = model(X)
#         loss = loss_fn(pred, y)
#
#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#         # Update class statistics
#         class_ids = y
#         for class_id in class_ids:
#             class_histogram_1[class_id] += 1
#
#         if step % ModelConfig.summary_write_interval == 0:
#             # Train loss
#             loss = loss.item()
#
#             # Train classification error.
#             label_pred = torch.argmax(pred, dim=1)
#             class_error = torch.sum(label_pred != y) / ModelConfig.batch_size
#
#             img = torchvision.utils.make_grid(X)
#             writer.add_image('Train/image', img, step)
#             writer.add_scalar('Train/loss', loss, step)
#
#             # Monitor the balanced random sampling
#             a = min(enumerate(class_histogram_1), key=lambda v: v[1])
#             b = max(enumerate(class_histogram_1), key=lambda v: v[1])
#             uniform_sampling_score = a[1] / b[1]
#
#             # Write for tensorboard.
#             writer.add_scalar('ClassBalancedStats/least_frequent_class',
#                               a[0], step)
#             writer.add_scalar('ClassBalancedStats/least_frequent_class_count',
#                               a[1], step)
#             writer.add_scalar('ClassBalancedStats/most_frequent_class',
#                               b[0], step)
#             writer.add_scalar('ClassBalancedStats/most_frequent_class count',
#                               b[1], step)
#             writer.add_scalar('ClassBalancedStats/uniform_sampling_score',
#                               uniform_sampling_score, step)
#             writer.add_scalar('Train/classification_error', class_error, step)
#
#             # Log on the console.
#             print("".join([
#                 f"[iter: {step:>5d}/{step_count:>5d}] ",
#                 f"loss: {loss:>7f}, ",
#                 f"classification_error: {class_error:>7f} "
#             ]))
#
# for epoch in range(10):
#     optimizer = torch.optim.Adam(resnet50.parameters())
#     train_loop(train_dataloader, resnet50, xent_loss, optimizer)
#     torch.save(resnet50.state_dict(), f'eth123_resnet50_{epoch}.pt')


def train_loop_v2(dataloader, model, loss_fn, na_loss_fn, optimizer,
                  writer, class_histogram_1):
    step_count = len(dataloader)
    model.train()

    for step, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update class statistics
        class_ids = y
        for class_id in class_ids:
            class_histogram_1[class_id] += 1

        if step != 0 and step % ModelConfig.summary_write_interval == 0:
            # Train loss
            loss = loss.item()

            # Train classification error.
            class_error = na_loss_fn(pred, y) / ModelConfig.batch_size

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


def main():
    # THE MODEL
    #
    # 1. A feature extractor model.
    reid_desc = ReidDescriptor50(dim=ModelConfig.reid_dim)
    # 2. A classifier model
    # resnet50 = torchvision.models.resnet50(num_classes=ETH123_DS.class_count)

    # THE LOSS FUNCTIONS
    #
    # xent_loss = torch.nn.CrossEntropyLoss()
    hm_loss = HypersphericManifoldLoss(ETH123_DS.class_count,
                                       ModelConfig.batch_size)
    na_loss = NearestAssignmentLoss(hm_loss.means)

    # THE DATASET
    train_dataset = ETH123_DS
    train_sample_gen = make_class_balanced_sampler(ETH123_DS)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sample_gen,
                                  batch_size=ModelConfig.batch_size)

    writer = SummaryWriter('train/eth123')
    class_histogram = [0] * ETH123_DS.class_count

    params_to_optimize = \
        list(reid_desc.parameters()) + \
        list(hm_loss.parameters())

    for epoch in range(10):
        optimizer = torch.optim.Adam(params_to_optimize)
        train_loop_v2(train_dataloader, reid_desc, hm_loss, na_loss,
                      optimizer, writer, class_histogram)

        torch.save(reid_desc.state_dict(), f'eth123_resnet50_{epoch}.pt')


if __name__ == "__main__":
   main()
