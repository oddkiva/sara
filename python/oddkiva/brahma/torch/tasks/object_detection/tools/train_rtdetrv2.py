# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import argparse
import atexit
import subprocess

from loguru import logger

import torch
import torch.nn.functional as F
from torch.distributed import (
    ReduceOp,
    destroy_process_group
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from oddkiva.brahma.torch import DEFAULT_DEVICE
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
from oddkiva.brahma.torch.object_detection.optim.ema import (
    ModelEMA,
    deparallelize
)
from oddkiva.brahma.torch.tasks.object_detection.configs.\
    train_config_rtdetrv2_r50vd_coco import (
        RTDETRv2,
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
    mb_used = [f'[GPU:{id}] {mb}' for id, mb in enumerate(mb_used)]
    mb_used = "\n".join(mb_used)
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
             anchor_class_logits) = aux_train_outputs['anchors']
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


def save_model(model: DDP | RTDETRv2,
               epoch: int,
               step: int | None = None) -> None:
    # Save the model after each training epoch.
    logger.debug(format_msg(f'Saving model at epoch {epoch}...'))
    if torchrun_is_running():
        # In the case of distributed training, make sure only the node
        # associated with GPU node 0 can save the model.
        if torch.distributed.get_rank() != 0:
            return

        assert isinstance(model, DDP)
        ckpt = model.module.state_dict()

        torch.save(
            ckpt,
            PipelineConfig.out_model_filepath(epoch, step)
        )
    else:
        assert isinstance(model, RTDETRv2)
        ckpt = model.state_dict()
        torch.save(
            ckpt,
            PipelineConfig.out_model_filepath(epoch, step)
        )


def train_for_one_epoch(
    dataloader: DataLoader,
    gpu_id: int | str | None,
    train_global_step: int,
    model: torch.nn.Module,
    loss_fn: RTDETRHungarianLoss,
    loss_reducer: HungarianLossReducer,
    optimizer: torch.optim.AdamW,
    ema: ModelEMA,
    writer: SummaryWriter,
    summary_write_interval: int,
    epoch: int,
    max_norm: float | None,
    debug: bool = False
) -> None:
    if debug:
        torch.autograd.set_detect_anomaly(True)

    model.train()

    for step, (imgs, tgt_boxes, tgt_labels) in enumerate(dataloader):
        # Optimize the GPU memory consumption as per the documentation.
        optimizer.zero_grad(set_to_none=True)

        if gpu_id is not None:
            imgs = imgs.to(gpu_id)
            tgt_boxes = [boxes_n.to(gpu_id) for boxes_n in tgt_boxes]
            tgt_labels = [labels_n.to(gpu_id) for labels_n in tgt_labels]

        logger.info(format_msg(
            f'[E:{epoch:0>2},S:{step:0>5}] Batch size: {imgs.shape[0]}'
        ))

        logger.info(format_msg(
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

        logger.info(format_msg(
            f'[E:{epoch:0>2},S:{step:0>5}] Calculating the Hungarian loss...'
        ))
        loss_dict = loss_fn.forward(
            box_geoms, box_class_logits,
            anchor_boxes, anchor_class_logits,
            dn_boxes, dn_class_logits, dn_groups,
            tgt_boxes, tgt_labels
        )

        logger.info(format_msg(
            f'[E:{epoch:0>2},S:{step:0>5}] Summing the elementary losses...'
        ))
        loss = loss_reducer.forward(loss_dict)
        logger.info(format_msg(
            f'[E:{epoch:0>2},S:{step:0>5}] Global loss = {loss}'
        ))

        logger.info(format_msg(
            f'[E:{epoch:0>2},S:{step:0>5}] Backpropagating...'
        ))
        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # AdamW and EMA should be used together.
        optimizer.step()
        ema.update(model)

        if step % summary_write_interval == 0:
            logger.info(format_msg(
                f'[E:{epoch:0>2},S:{step:0>5}] Logging to tensorboard...'
            ))

            log_elementary_losses(loss_dict, writer, train_global_step)

            loss_value = loss.detach()
            if torchrun_is_running():
                torch.distributed.all_reduce(loss_value, ReduceOp.AVG);
            writer.add_scalar(f'global', loss_value, train_global_step)

            if torch.cuda.is_available():
                logger.info(format_msg((
                    f'[E:{epoch:0>2},S:{step:0>5}] Memory usage:\n'
                    f'{get_cuda_memory_usage()}'
                )))

        train_global_step += 1

        if step > 0 and step % 1000 == 0:
            save_model(model, epoch, step)


def main(args):
    # PARALLEL TRAINING
    ddp_setup()


    # --------------------------------------------------------------------------
    # THE MODEL
    gpu_id = get_local_rank() if torchrun_is_running() else DEFAULT_DEVICE
    rtdetrv2_model = PipelineConfig.make_model()
    # Load the model weights.
    ckpt_fp = args.resume
    if ckpt_fp is not None:
        logger.info(format_msg(
            f"Loading model weights from checkpoint: {ckpt_fp}"
        ))
        ckpt = torch.load(ckpt_fp, map_location='cpu')

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
    else:
        rtdetrv2_model = rtdetrv2_model.to(gpu_id)


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
    rtdetrv2_param_groups = deparallelize(rtdetrv2_model).\
        group_learnable_parameters()
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
