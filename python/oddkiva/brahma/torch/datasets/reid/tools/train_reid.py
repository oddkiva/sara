import torch
import torch.nn
import torchvision
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.datasets.reid.iust_person_reid import IUSTPersonReID
from oddkiva.brahma.torch.datasets.reid.triplet_dataset import (
    TripletDataset
)
from oddkiva.brahma.torch.datasets.reid.models.configs import (
    ReID_Resnet50_256 as ModelConfig
)
from oddkiva.brahma.torch.datasets.reid.triplet_loss import TripletLoss


# Dataset
DATASET_DIR_PATH = DATA_DIR_PATH / 'reid' / 'IUSTPersonReID'
assert DATASET_DIR_PATH.exists()


def validate(
    dataloader: DataLoader, test_global_step: int,
    model: torch.nn.Module, writer: SummaryWriter
) -> None:
    model.eval()

    step_count = len(dataloader)

    with torch.no_grad():
        for step, (X, _) in enumerate(dataloader):
            anchor, pos, neg = X
            d_anchor, d_pos, d_neg = model(anchor), model(pos), model(neg)
            dist_ap = torch.norm(d_anchor - d_pos)
            dist_an = torch.norm(d_anchor - d_neg)

            if step % ModelConfig.summary_write_interval == 0:
                img_anchor = torchvision.utils.make_grid(anchor)
                img_pos = torchvision.utils.make_grid(pos)
                img_neg = torchvision.utils.make_grid(neg)
                writer.add_image('Val/anchors', img_anchor, test_global_step)
                writer.add_image('Val/positives', img_pos, test_global_step)
                writer.add_image('Val/negatives', img_neg, test_global_step)
                writer.add_scalar('Val/dist_ap', dist_ap, test_global_step)
                writer.add_scalar('Val/dist_an', dist_an, test_global_step)

                # Log on the console.
                print("".join([
                    f"[test_global_step: {test_global_step:>5d}]",
                    f"[iter: {step:>5d}/{step_count:>5d}] ",
                    f"dist_ap: {dist_ap:>7f}  "
                    f"dist_an: {dist_an:>7f}"
                ]))

                test_global_step += 1


def train_for_one_epoch(
    dataloader: DataLoader, train_global_step: int,
    model: torch.nn.Module,
    triplet_loss: TripletLoss, optimizer: torch.optim.Optimizer,
    writer: SummaryWriter, class_histogram_1
) -> None:
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

        if step % ModelConfig.summary_write_interval == 0:
            # Train loss
            loss = loss.item()

            img_anchor = torchvision.utils.make_grid(anchor)
            img_pos = torchvision.utils.make_grid(pos)
            img_neg = torchvision.utils.make_grid(neg)
            writer.add_image('Train/anchors', img_anchor, train_global_step)
            writer.add_image('Train/positives', img_pos, train_global_step)
            writer.add_image('Train/negatives', img_neg, train_global_step)
            writer.add_scalar('Train/triplet_loss', loss, train_global_step)

            # Monitor the balanced random sampling
            a = min(enumerate(class_histogram_1), key=lambda v: v[1])
            b = max(enumerate(class_histogram_1), key=lambda v: v[1])
            uniform_sampling_score = a[1] / b[1]

            # Write for tensorboard.
            writer.add_scalar('ClassBalancedStats/least_frequent_class',
                              a[0], train_global_step)
            writer.add_scalar('ClassBalancedStats/least_frequent_class_count',
                              a[1], train_global_step)
            writer.add_scalar('ClassBalancedStats/most_frequent_class',
                              b[0], train_global_step)
            writer.add_scalar('ClassBalancedStats/most_frequent_class count',
                              b[1], train_global_step)
            writer.add_scalar('ClassBalancedStats/uniform_sampling_score',
                              uniform_sampling_score, train_global_step)

            # Log on the console.
            print("".join([
                f"[train_global_step: {train_global_step:>5d}]",
                f"[iter: {step:>5d}/{step_count:>5d}] ",
                f"triplet_loss: {loss:>7f}"
            ]))

            train_global_step += 1

def main():
    # THE DATASET
    data_transform = v2.Compose([
        v2.Resize(ModelConfig.image_size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])
    train_ds = IUSTPersonReID(DATASET_DIR_PATH,
                              transform=data_transform,
                              dataset_type='train')
    test_ds = IUSTPersonReID(DATASET_DIR_PATH,
                             transform=data_transform,
                             dataset_type='test')

    writer = SummaryWriter(ModelConfig.summary_out_dir)
    class_histogram = [0] * train_ds.class_count

    # THE MODEL
    reid_model = ModelConfig.make()

    # THE LOSS FUNCTION
    triplet_loss = TripletLoss(
        summary_writer=writer,
        summary_write_interval=ModelConfig.summary_write_interval
    )

    train_global_step = 0
    test_global_step = 0
    for epoch in range(10):
        # Restart the state of the Adam optimizer every epoch.
        optimizer = torch.optim.Adam(reid_model.parameters())

        # Resample the list of triplets for each epoch.
        train_tds = TripletDataset(train_ds)
        train_dataloader = DataLoader(train_tds, batch_size=ModelConfig.batch_size)

        # Train the model.
        train_for_one_epoch(train_dataloader, train_global_step,
                   reid_model, triplet_loss, optimizer,
                   writer, class_histogram)

        # Save the model after each training epoch.
        torch.save(reid_model.state_dict(), f'eth123_resnet50_{epoch}.pt')

        # Evaluate the model.
        test_tds = TripletDataset(test_ds)
        test_dataloader = DataLoader(test_tds, batch_size=1)
        validate(test_dataloader, test_global_step,
                 reid_model, writer)



if __name__ == "__main__":
    main()
