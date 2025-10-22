import atexit
import logging
from rich.logging import RichHandler

import torch
import torch.nn
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from oddkiva.brahma.torch.logging import FORMAT
from oddkiva.brahma.torch.parallel.ddp import ddp_setup, get_local_rank
from oddkiva.brahma.torch.datasets.reid.triplet_loss import TripletLoss
import oddkiva.brahma.torch.tasks.reid.configs as C


# logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]"
)
LOGGER = logging.getLogger(__name__)

CONFIGS = {
    'ethz': C.Ethz,
    'ethz_variant': C.EthzVariant,
    'iust': C.Iust
}

PipelineConfig = CONFIGS['ethz']


# --------------------------------------------------------------------------
# PARALLEL TRAINING
# Automatically clean up the parallel training environment.
# --------------------------------------------------------------------------
@atexit.register
def ddp_cleanup():
    local_rank = get_local_rank()
    rank = get_rank()
    print(f'[DDP][rank:{rank}][local_rank:{local_rank}] CLEANUP')
    destroy_process_group()


def validate(
    dataloader: DataLoader, gpu_id: int, test_global_step: int,
    model: torch.nn.Module, triplet_loss: torch.nn.Module,
    writer: SummaryWriter, summary_write_interval: int
) -> None:
    model.eval()

    step_count = len(dataloader)

    with torch.no_grad():
        for step, (X, _) in enumerate(dataloader):
            anchor, pos, neg = X

            # Transfer the data to the appropriate GPU node.
            anchor1 = anchor.to(gpu_id)
            pos1 = pos.to(gpu_id)
            neg1 = neg.to(gpu_id)

            d_anchor, d_pos, d_neg = model(anchor1), model(pos1), model(neg1)
            dist_ap = torch.mean(torch.sum((d_anchor - d_pos) ** 2, dim=-1))
            dist_an = torch.mean(torch.sum((d_anchor - d_neg) ** 2, dim=-1))
            loss = triplet_loss(d_anchor, d_pos, d_neg,
                                [*model.parameters()])

            if step % summary_write_interval == 0:
                img_anchor = torchvision.utils.make_grid(anchor)
                img_pos = torchvision.utils.make_grid(pos)
                img_neg = torchvision.utils.make_grid(neg)
                writer.add_image('Val/anchors', img_anchor, test_global_step)
                writer.add_image('Val/positives', img_pos, test_global_step)
                writer.add_image('Val/negatives', img_neg, test_global_step)
                writer.add_scalar('Val/dist_ap', dist_ap, test_global_step)
                writer.add_scalar('Val/dist_an', dist_an, test_global_step)
                writer.add_scalar('Val/triplet_loss', loss, test_global_step)

                # Log on the console.
                LOGGER.info("".join([
                    f"[test_global_step: {test_global_step:>5d}]",
                    f"[iter: {step:>5d}/{step_count:>5d}] ",
                    f"dist_ap: {dist_ap:>7f}  "
                    f"dist_an: {dist_an:>7f}"
                ]))

                test_global_step += 1

def train_for_one_epoch(
    dataloader: DataLoader,
    gpu_id: int,
    train_global_step: int,
    model: torch.nn.Module,
    triplet_loss: TripletLoss, optimizer: torch.optim.Optimizer,
    writer: SummaryWriter, summary_write_interval: int,
    class_histogram_1
) -> None:
    torch.autograd.set_detect_anomaly(True)

    step_count = len(dataloader)
    model.train()

    for step, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()

        anchor, pos, neg = X

        # Transfer the data to the appropriate GPU node.
        anchor = anchor.to(gpu_id)
        pos = pos.to(gpu_id)
        neg = neg.to(gpu_id)

        d_anchor, d_pos, d_neg = model(anchor), model(pos), model(neg)
        loss = triplet_loss(d_anchor, d_pos, d_neg, [*model.parameters()])

        loss.backward()
        optimizer.step()

        # Update class statistics
        for yi in y:
            class_ids = yi
            for class_id in class_ids:
                class_histogram_1[class_id] += 1

        if gpu_id == 0 and step % summary_write_interval == 0:
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
            LOGGER.info("".join([
                f"[train_global_step: {train_global_step:>5d}]",
                f"[iter: {step:>5d}/{step_count:>5d}] ",
                f"triplet_loss: {loss:>7f}"
            ]))

            train_global_step += 1


def main():
    # --------------------------------------------------------------------------
    # PARALLEL TRAINING
    # --------------------------------------------------------------------------
    ddp_setup()

    # THE DATASET
    train_ds, val_ds, _ = PipelineConfig.make_datasets()
    writer = PipelineConfig.make_summary_writer()
    class_histogram = [0] * train_ds.class_count

    # THE MODEL
    #
    # --------------------------------------------------------------------------
    # PARALLEL TRAINING
    #
    # 1. Ensure that we transfer the model weights to the correct GPU node.
    # --------------------------------------------------------------------------
    gpu_id = get_local_rank()
    monogpu_reid_model = PipelineConfig.make_model().to(gpu_id)
    # 2. Make the batch normalization layers parallel-friendly
    monogpu_reid_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        monogpu_reid_model
    )
    reid_model = DDP(
        monogpu_reid_model,
        device_ids=[gpu_id],
    )

    # THE LOSS FUNCTION
    triplet_loss = TripletLoss(
        summary_writer=writer,
        summary_write_interval=PipelineConfig.write_interval
    )
    triplet_loss.to(gpu_id)
    # There is no model weights on the triplet loss, so the following is not
    # necessary:

    train_global_step = 0
    val_global_step = 0
    for epoch in range(10):
        LOGGER.info(f"learning rate = {PipelineConfig.learning_rate}",  extra={"markup": True})
        # Restart the state of the Adam optimizer every epoch.
        optimizer = torch.optim.Adam(reid_model.parameters(),
                                     PipelineConfig.learning_rate)

        # Resample the list of triplets for each epoch.
        train_dl = PipelineConfig.make_triplet_dataset(train_ds)
        train_dl.sampler.set_epoch(epoch)   # call this additional line at every epoch

        # Train the model.
        train_for_one_epoch(train_dl, gpu_id,
                            train_global_step,
                            reid_model, triplet_loss, optimizer,
                            writer, PipelineConfig.write_interval,
                            class_histogram)

        # Save the model after each training epoch.
        # Only the node associated with GPU node 0 can save the model.
        if gpu_id == 0:
            ckpt = reid_model.module.state_dict()
            torch.save(
                ckpt,
                PipelineConfig.out_model_filepath(epoch)
            )

        # Evaluate the model.
        if gpu_id == 0:
            val_dl = PipelineConfig.make_triplet_dataset(val_ds)
            val_dl.sampler.set_epoch(epoch);
            validate(val_dl, gpu_id, val_global_step, reid_model, triplet_loss,
                     writer, PipelineConfig.write_interval)


if __name__ == "__main__":
    main()
