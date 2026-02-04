# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import argparse
import atexit
import subprocess
import sys

from loguru import logger

import torch
from torch.distributed import (
    ReduceOp,
    destroy_process_group
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from oddkiva.brahma.torch.utils.freeze import (
    freeze_batch_norm,
    freeze_parameters
)
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


def get_cuda_memory_usage():
    result = subprocess.run(
        ['nvidia-smi',
         '--query-gpu=memory.used',
         '--format=csv,noheader'],
        stdout=subprocess.PIPE
    )
    mb_used = result.stdout.decode('utf-8').strip().split('\n')
    mb_used = [f'/GPU/{id}:{mb}' for id, mb in enumerate(mb_used)]
    mb_used = ", ".join(mb_used)
    return mb_used


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
    model: torch.nn.Module,
    loss_fn: RTDETRHungarianLoss,
    loss_reducer: HungarianLossReducer,
    optimizer: torch.optim.AdamW,
    ema: ModelEMA,
    writer: SummaryWriter,
    summary_write_interval: int,
    epoch: int,
    max_norm: float | None,
    debug: bool = False,
) -> None:
    if debug:
        torch.autograd.set_detect_anomaly(True)

    model.train()

    for step, (imgs, tgt_boxes, tgt_labels) in enumerate(dataloader):
        train_global_step = epoch * len(dataloader) + step

        # Optimize the GPU memory consumption as per the documentation.
        optimizer.zero_grad(set_to_none=True)

        if gpu_id is not None:
            imgs = imgs.to(gpu_id)
            tgt_boxes = [boxes_n.to(gpu_id) for boxes_n in tgt_boxes]
            tgt_labels = [labels_n.to(gpu_id) for labels_n in tgt_labels]
        logger.trace(format_msg(
            f'[E:{epoch:0>2},S:{step:0>5}] Batch size: {imgs.shape[0]}'
        ))

        logger.trace(format_msg(
            f'[E:{epoch:0>2},S:{step:0>5}] Feeding annotated images...'
        ))
        targets = {
            'boxes': tgt_boxes,
            'labels': tgt_labels
        }
        box_geoms, box_class_logits, aux_train_outputs = model.forward(
            imgs, targets
        )

        anchor_boxes, anchor_class_logits = aux_train_outputs['anchors']
        dn_boxes, dn_class_logits = aux_train_outputs['dn_boxes']
        dn_groups = aux_train_outputs['dn_groups']

        logger.trace(format_msg(
            f'[E:{epoch:0>2},S:{step:0>5}] Calculating the Hungarian loss...'
        ))
        loss_dict = loss_fn.forward(
            box_geoms, box_class_logits,
            anchor_boxes, anchor_class_logits,
            dn_boxes, dn_class_logits, dn_groups,
            tgt_boxes, tgt_labels
        )

        logger.trace(format_msg(
            f'[E:{epoch:0>2},S:{step:0>5}] Summing the elementary losses...'
        ))
        loss = loss_reducer.forward(loss_dict)
        logger.info(format_msg(
            f'[E:{epoch:0>2},S:{step:0>5}] loss = {loss:9.6f}'
        ))

        logger.trace(format_msg(
            f'[E:{epoch:0>2},S:{step:0>5}] Backpropagating...'
        ))
        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # AdamW and EMA should be used together.
        optimizer.step()
        ema.update(model)

        if step % summary_write_interval == 0:
            # NOTE:
            # 1. Calculate the average loss across all GPUs.
            # 2. Log only on the CPU process using GPU node #0.
            logger.trace(format_msg(
                f'[E:{epoch:0>2},S:{step:0>5}] Logging to tensorboard...'
            ))

            log_elementary_losses(loss_dict, writer, train_global_step)

            loss_value = loss.detach()
            torch.distributed.all_reduce(loss_value, ReduceOp.AVG);
            if torchrun_is_running() and torch.distributed.get_rank() == 0:
                writer.add_scalar(f'global', loss_value, train_global_step)

            if (torchrun_is_running() and
                    torch.distributed.get_rank() == 0 and
                    torch.cuda.is_available()):
                logger.info(f'[CUDA] {get_cuda_memory_usage()}')

        if step > 0 and step % 1000 == 0:
            save_model(model, epoch, step)


def validate(
    dataloader: DataLoader,
    gpu_id: int | None,
    epoch: int,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loss_reducer: HungarianLossReducer,
    writer: SummaryWriter,
) -> None:
    model.eval()

    n = len(dataloader)
    ws = torch.distributed.get_world_size()
    rk = torch.distributed.get_rank()

    with torch.no_grad():
        for step, (imgs, tgt_boxes, tgt_labels) in enumerate(dataloader):
            val_global_step = n * ws * epoch + ws * step + rk

            if gpu_id is not None:
                imgs = imgs.to(gpu_id)
                tgt_boxes = [boxes_n.to(gpu_id) for boxes_n in tgt_boxes]
                tgt_labels = [labels_n.to(gpu_id) for labels_n in tgt_labels]

            logger.trace(format_msg((
                f"[V][E:{epoch:0>2},S:{step:0>5}] "
                "Feeding annotated images..."
            )))
            targets = {
                'boxes': tgt_boxes,
                'labels': tgt_labels
            }
            box_geoms, box_class_logits, aux_train_outputs = model.forward(
                imgs, targets
            )

            anchor_boxes, anchor_class_logits = aux_train_outputs['anchors']
            dn_boxes, dn_class_logits = aux_train_outputs['dn_boxes']
            dn_groups = aux_train_outputs['dn_groups']

            logger.trace(format_msg((
                f"[V][E:{epoch:0>2},S:{step:0>5}] "
                "Calculating the Hungarian loss..."
            )))
            loss_dict = loss_fn.forward(
                box_geoms, box_class_logits,
                anchor_boxes, anchor_class_logits,
                dn_boxes, dn_class_logits, dn_groups,
                tgt_boxes, tgt_labels
            )

            logger.trace(format_msg((
                f"[V][E:{epoch:0>2},S:{step:0>5}] "
                "Summing the elementary losses..."
            )))
            loss = loss_reducer.forward(loss_dict)
            logger.info(format_msg(
                f"[V][E:{epoch:0>2},S:{step:0>5}] Loss = {loss:9.6f}"
            ))

            logger.trace(format_msg(
                f"[V][E:{epoch:0>2},S:{step:0>5}] Logging to tensorboard..."
            ))

            logger.debug(format_msg((
                f"[V][E:{epoch:0>2},S:{step:0>5}] "
                f"global step: {val_global_step}"
            )))
            writer.add_scalar(f'val/global', loss, val_global_step)

    if torchrun_is_running():
        logger.debug(format_msg((
            f"[V][E:{epoch:0>2}] "
            "Waiting for validation completion across all nodes..."
        )))
        torch.distributed.barrier()


def main(args):
    if args.log_level is not None:
        logger.add(sys.stdout, level=str(args.log_level).upper())

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
        ckpt = torch.load(ckpt_fp, map_location='cpu', weights_only=True)

        # NOTE:
        # Clean up the checkpoint file as we fixed the implementation of the
        # transformer decoder recently.
        ckpt = {
            k: v
            for k, v in ckpt.items()
            if (not k.startswith('decoder.decoder_class_logits_head') and
                not k.startswith('decoder.decoder_box_geometry_head'))
        }

        rtdetrv2_model.load_state_dict(ckpt)

    if args.freeze_low_layers:
        # NOTE:
        # In later epochs, we can freeze the parameters of:
        # - the first block of the backbone
        # - the batch norm layers of the backbone
        # This enables to free up a lot of GPU memory and increase the batch
        # size from 5 to 8.
        #
        # This is what RT-DETR's original implementation does and it can afford
        # to do that as it starts from a pretrained backbone.
        freeze_batch_norm(rtdetrv2_model.backbone)
        freeze_parameters(rtdetrv2_model.backbone.blocks[0])

    # Transfer the model to GPU memory and wrap it as a DDP model.
    if torchrun_is_running():
        rtdetrv2_model = wrap_model_with_ddp_if_needed(rtdetrv2_model)
        torch.distributed.barrier()


    # --------------------------------------------------------------------------
    # THE LOSS FUNCTION
    classification_loss_params = {
        # The alpha parameter is a parameter of the **VARIFOCAL** loss.
        # It is large and should gives a very strong emphasis on minimizing
        # false positives...
        'alpha': 0.75,
        'gamma': 2.0
    }
    box_matcher_params = {
        # The alpha parameter is a parameter of the **FOCAL** loss.
        #
        # It gives less emphasis on the cost incurred in the positive
        # part of the focal loss. The negative part of the focal loss has more
        # importance.
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
    for epoch in range(10):
        logger.info(format_msg(
            f"learning rate = {PipelineConfig.learning_rate}"
        ))

        # Get the train dataloader.
        train_dl = PipelineConfig.make_train_dataloader(train_ds)
        if torchrun_is_running():
            # NOTE: ensure the shuffling is different at each epoch in
            # distributed mode (cf.
            # https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)
            assert type(train_dl.sampler) is torch.utils.data.DistributedSampler
            train_dl.sampler.set_epoch(epoch)

        # Train the model.
        train_for_one_epoch(train_dl, gpu_id,
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
        validate(val_dl, gpu_id, epoch,
                 rtdetrv2_model, hungarian_loss_fn, loss_reducer,
                 summary_writer)


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
    parser.add_argument(
        '-f', '--freeze_low_layers',
        action='store_true',
        help="Freeze the low level layers of the backbone"
    )
    parser.add_argument(
        '-l', '--log_level',
        type=str,
        help="Logging level [trace|debug|info]"
    )
    args = parser.parse_args()

    if args.resume and args.backbone:
        print(("ERROR: choose either to resume from an existing checkpoint or "
               "to load the backbone weights"))
        exit()

    main(args)
