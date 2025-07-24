import platform
from pathlib import Path

import torch
import torch.nn
import torchvision
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from oddkiva.brahma.torch.datasets.reid.eth123 import ETH123
from oddkiva.brahma.torch.datasets.reid.triplet_dataset import (
    TripletDatabase
)
from oddkiva.brahma.torch.datasets.reid.triplet_loss import TripletLoss


class ModelConfig:
    reid_dim = 256
    image_size = (160, 64)
    batch_size = 32
    summary_write_interval = 1

    # Add distributed training configs
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # rank = int(os.environ.get("RANK", 0))


# Dataset
if platform.system() == 'Darwin':
    ROOT_PATH = Path('/Users/oddkiva/Downloads/reid/dataset_ETHZ/')
else:
    ROOT_PATH = Path('/home/david/GitLab/oddkiva/sara/data/reid/dataset_ETHZ/')
# Data transform
DATA_TRANSFORM = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(ModelConfig.image_size, antialias=True)
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


def train_loop(dataloader: DataLoader, model: torch.nn.Module,
               triplet_loss: TripletLoss, optimizer: torch.optim.Optimizer,
               writer: SummaryWriter, class_histogram_1):
    step_count = len(dataloader)
    model.train()

    for step, (X, y) in enumerate(dataloader):
        anchor, pos, neg = X
        d_anchor, d_pos, d_neg = model(anchor), model(pos), model(neg)
        loss = triplet_loss.forward(d_anchor, d_pos, d_neg,
                                    [*model.parameters()])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update class statistics
        for yi in y:
            class_ids = yi
            for class_id in class_ids:
                class_histogram_1[class_id] += 1

        if step != 0 and step % ModelConfig.summary_write_interval == 0:
            # Train loss
            loss = loss.item()

            img_anchor = torchvision.utils.make_grid(anchor)
            img_pos = torchvision.utils.make_grid(pos)
            img_neg = torchvision.utils.make_grid(neg)
            writer.add_image('Train/anchors', img_anchor, step)
            writer.add_image('Train/positives', img_pos, step)
            writer.add_image('Train/negatives', img_neg, step)
            writer.add_scalar('Train/triplet_loss', loss, step)

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

            # Log on the console.
            print("".join([
                f"[iter: {step:>5d}/{step_count:>5d}] ",
                f"triplet_loss: {loss:>7f}, "
            ]))


def main():
    # THE DATASET
    train_dataset = ETH123_DS
    writer = SummaryWriter('train/eth123')
    class_histogram = [0] * ETH123_DS.class_count

    # THE MODEL
    #
    # 1. A feature extractor model.
    reid_desc = ReidDescriptor50(dim=ModelConfig.reid_dim)
    # 2. A classifier model
    # resnet50 = torchvision.models.resnet50(num_classes=ETH123_DS.class_count)

    # THE LOSS FUNCTIONS
    #
    triplet_loss = TripletLoss(
        summary_writer=writer,
        summary_write_interval=ModelConfig.summary_write_interval
    )

    for epoch in range(10):
        # Restart the state of the Adam optimizer every epoch.
        optimizer = torch.optim.Adam(reid_desc.parameters())

        # Resample the list of triplets for each epoch.
        train_tds = TripletDatabase(train_dataset)
        train_dataloader = DataLoader(train_tds, batch_size=ModelConfig.batch_size)

        train_loop(train_dataloader, reid_desc, triplet_loss, optimizer,
                   writer, class_histogram)

        torch.save(reid_desc.state_dict(), f'eth123_resnet50_{epoch}.pt')


if __name__ == "__main__":
    main()
