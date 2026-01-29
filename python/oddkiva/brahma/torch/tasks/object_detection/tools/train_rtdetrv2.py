# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import argparse
import atexit
from pathlib import Path

from loguru import logger

import torch
import torch.nn.functional as F
from torch.distributed import (
    ReduceOp,
    destroy_process_group
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from oddkiva.brahma.torch.utils.logging import format_msg
from oddkiva.brahma.torch.parallel.ddp import (
    ddp_setup,
    get_local_rank,
    torchrun_is_running,
    wrap_model_with_ddp_if_needed
)
from oddkiva.brahma.torch.object_detection.detr.architectures\
    .rt_detr.hungarian_loss import (
        HungarianLossReducer,
        RTDETRHungarianLoss,
        log_elementary_losses
    )
from oddkiva.brahma.torch.object_detection.optim.ema import ModelEMA
from oddkiva.brahma.torch.tasks.object_detection.configs.\
    train_config_rtdetrv2_r50vd_coco import (
        TrainTestPipelineConfig as PipelineConfig
    )


# --------------------------------------------------------------------------
# PARALLEL TRAINING
# Automatically clean up the parallel training environment.
# --------------------------------------------------------------------------
@atexit.register
def ddp_cleanup():
    if not torchrun_is_running():
        return
    logger.info(format_msg("Cleaning DistributedDataParallel environment..."))
    destroy_process_group()


def validate(
    dataloader: DataLoader,
    gpu_id: int | None,
    val_global_step: int,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loss_reducer: HungarianLossReducer,
    writer: SummaryWriter,
    summary_write_interval: int
) -> None:
    model.eval()

    with torch.no_grad():
        for step, (imgs, tgt_boxes, tgt_labels) in enumerate(dataloader):
            if gpu_id is not None:
                imgs = imgs.to(gpu_id)
                tgt_boxes = [boxes_n.to(gpu_id) for boxes_n in tgt_boxes]
                tgt_labels = [labels_n.to(gpu_id) for labels_n in tgt_labels]

            logger.info(format_msg(f'[val][step:{step}] Feeding annotated images...'))
            targets = {
                'boxes': tgt_boxes,
                'labels': tgt_labels
            }
            box_geoms, box_class_logits, aux_train_outputs = model.forward(
                imgs, targets
            )

            (anchor_geometry_logits,
             anchor_class_logits) = aux_train_outputs['top_k_anchor_boxes']
            anchor_boxes = F.sigmoid(anchor_geometry_logits)
            dn_boxes, dn_class_logits = aux_train_outputs['dn_boxes']
            dn_groups = aux_train_outputs['dn_groups']

            logger.info(format_msg(f'[val][step:{step}] Calculating the Hungarian loss...'))
            loss_dict = loss_fn.forward(
                box_geoms, box_class_logits,
                anchor_boxes, anchor_class_logits,
                dn_boxes, dn_class_logits, dn_groups,
                tgt_boxes, tgt_labels
            )

            logger.info(format_msg(f'[val][step:{step}] Summing the elementary losses...'))
            loss = loss_reducer.forward(loss_dict)
            logger.info(format_msg(f'[val][step:{step}] Global loss = {loss}'))

            if step % summary_write_interval == 0:
                logger.info(format_msg(f'[val][step:{step}] Logging to tensorboard...'))
                loss_value = loss
                torch.distributed.all_reduce(loss_value, ReduceOp.AVG);
                writer.add_scalar(f'val/global', loss_value, val_global_step)

            val_global_step += 1


def save_model(rtdetrv2_model: torch.nn.Module,
               epoch: int,
               step: int | None = None) -> None:
    # Save the model after each training epoch.
    if torch.distributed.get_rank() == 0 and torchrun_is_running():
        logger.debug(format_msg(f'Saving model at epoch {epoch}...'))
        assert isinstance(rtdetrv2_model,
                          torch.nn.parallel.DistributedDataParallel)
        # In the case of distributed training, make sure only the node
        # associated with GPU node 0 can save the model.
        ckpt = rtdetrv2_model.module.state_dict()
        torch.save(
            ckpt,
            PipelineConfig.out_model_filepath(epoch, step)
        )


def train_for_one_epoch(
    dataloader: DataLoader,
    gpu_id: int | None,
    train_global_step: int,
    model: torch.nn.Module,
    loss_fn: RTDETRHungarianLoss,
    loss_reducer: HungarianLossReducer,
    optimizer: torch.optim.AdamW,
    ema: ModelEMA,
    writer: SummaryWriter,
    summary_write_interval: int,
    epoch: int,
    max_norm: float | None
) -> None:
    torch.autograd.set_detect_anomaly(True)

    model.train()

    for step, (imgs, tgt_boxes, tgt_labels) in enumerate(dataloader):
        optimizer.zero_grad()

        if gpu_id is not None:
            imgs = imgs.to(gpu_id)
            tgt_boxes = [boxes_n.to(gpu_id) for boxes_n in tgt_boxes]
            tgt_labels = [labels_n.to(gpu_id) for labels_n in tgt_labels]

        logger.info(format_msg(f'[E:{epoch:0>2},S:{step:0>5}] Feeding annotated images...'))
        targets = {
            'boxes': tgt_boxes,
            'labels': tgt_labels
        }
        box_geoms, box_class_logits, aux_train_outputs = model.forward(
            imgs, targets
        )

        (anchor_geometry_logits,
         anchor_class_logits) = aux_train_outputs['top_k_anchor_boxes']
        anchor_boxes = F.sigmoid(anchor_geometry_logits)
        dn_boxes, dn_class_logits = aux_train_outputs['dn_boxes']
        dn_groups = aux_train_outputs['dn_groups']

        logger.info(format_msg(f'[E:{epoch:0>2},S:{step:0>5}] Calculating the Hungarian loss...'))
        loss_dict = loss_fn.forward(
            box_geoms, box_class_logits,
            anchor_boxes, anchor_class_logits,
            dn_boxes, dn_class_logits, dn_groups,
            tgt_boxes, tgt_labels
        )

        logger.info(format_msg(f'[E:{epoch:0>2},S:{step:0>5}] Summing the elementary losses...'))
        loss = loss_reducer.forward(loss_dict)
        logger.info(format_msg(f'[E:{epoch:0>2},S:{step:0>5}] Global loss = {loss}'))

        logger.info(format_msg(f'[E:{epoch:0>2},S:{step:0>5}] Backpropagating...'))
        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # AdamW and EMA should be used together.
        optimizer.step()
        ema.update(model)

        if step % summary_write_interval == 0:
            logger.info(format_msg(f'[E:{epoch:0>2},S:{step:0>5}] Logging to tensorboard...'))

            log_elementary_losses(loss_dict, writer, train_global_step)

            loss_value = loss
            torch.distributed.all_reduce(loss_value, ReduceOp.AVG);
            writer.add_scalar(f'global', loss_value, train_global_step)

        train_global_step += 1

        if step > 0 and step % 1000 == 0:
            save_model(model, epoch, step)


def main(args):
    # PARALLEL TRAINING
    ddp_setup()

    # --------------------------------------------------------------------------
    # THE MODEL
    gpu_id = get_local_rank()
    rtdetrv2_model = PipelineConfig.make_model()
    # Load the model weights.
    ckpt_fp = args.resume
    if ckpt_fp is not None:
        logger.info(format_msg(
            f"Loading model weights from checkpoint: {ckpt_fp}"
        ))
        ckpt = torch.load(ckpt_fp, map_location='cpu')
        rtdetrv2_model.load_state_dict(ckpt)
    # Transfer the model to GPU memory and wrap it as a DDP model.
    rtdetrv2_model = wrap_model_with_ddp_if_needed(rtdetrv2_model)
    torch.distributed.barrier()


    # --------------------------------------------------------------------------
    # THE LOSS FUNCTION
    classification_loss_params = {
        'alpha': 0.75,
        'gamma': 2.0
    }
    box_matcher_params = {
        'alpha': 0.25,
        'gamma': 2.0,
        'cost_matrix_weights': {
            'class': 2.0,
            'l1': 5.0,
            'giou': 2.0
        }
    }
    hungarian_loss_fn = RTDETRHungarianLoss(
        classification_loss_params=classification_loss_params,
        box_matcher_params=box_matcher_params

    )

    # --------------------------------------------------------------------------
    # THE COMPOUND LOSS FUNCTION
    #
    # The weights of each elementary losses.
    loss_weights = {
        'vf': 1.0,
        'l1': 5.0,
        'giou': 2.0
    }
    loss_reducer = HungarianLossReducer(loss_weights)


    # --------------------------------------------------------------------------
    # THE OPTIMIZER.
    #
    # The parameter groups with specific learning parameters.
    rtdetrv2_param_groups = rtdetrv2_model.module.group_learnable_parameters()
    # We learn from scratch: let's be very aggressive.
    backbone_pg = rtdetrv2_param_groups[0]
    backbone_pg['lr'] = 5e-5
    # The optimizer.
    adamw = torch.optim.AdamW(rtdetrv2_param_groups,
                              lr=PipelineConfig.learning_rate,
                              betas=PipelineConfig.betas,
                              weight_decay=PipelineConfig.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        adamw,
        milestones=[1000],
        gamma=0.1
    )

    ema = ModelEMA(rtdetrv2_model,
                   decay=PipelineConfig.ema_decay,
                   warmups=PipelineConfig.ema_warmup_steps)


    # --------------------------------------------------------------------------
    # THE DATA.
    train_ds, val_ds, _ = PipelineConfig.make_datasets()
    summary_writer = PipelineConfig.make_summary_writer()


    # --------------------------------------------------------------------------
    # TRAIN AND VALIDATE.
    train_global_step = 0
    val_global_step = 0
    for epoch in range(10):
        logger.info(format_msg(
            f"learning rate = {PipelineConfig.learning_rate}"
        ))

        # Get the train dataloader.
        train_dl = PipelineConfig.make_train_dataloader(train_ds)
        if torchrun_is_running():
            train_dl.sampler.set_epoch(epoch)

        # Train the model.
        train_for_one_epoch(train_dl, gpu_id,
                            train_global_step,
                            rtdetrv2_model,
                            hungarian_loss_fn,
                            loss_reducer,
                            adamw,
                            ema,
                            summary_writer,
                            PipelineConfig.write_interval,
                            epoch,
                            PipelineConfig.gradient_norm_max)

        # Modulate the learning rate after each epoch.
        lr_scheduler.step()

        # Save the model after each training epoch.
        save_model(rtdetrv2_model, epoch)

        # Evaluate the model.
        val_dl = PipelineConfig.make_val_dataloader(val_ds)
        if torchrun_is_running():
            val_dl.sampler.set_epoch(epoch)
        validate(val_dl, gpu_id, val_global_step,
                 rtdetrv2_model,
                 hungarian_loss_fn, loss_reducer,
                 summary_writer,
                 PipelineConfig.write_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-r', '--resume',
        type=str,
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '-b', '--backbone',
        type=str,
        help=("Load the backbone weights from the public checkpoint provided "
              "by RT-DETR's authors.")
    )
    args = parser.parse_args()

    if args.resume and args.backbone:
        print(("ERROR: choose either to resume from an existing checkpoint or "
               "to load the backbone weights"))
        exit()

    main(args)
