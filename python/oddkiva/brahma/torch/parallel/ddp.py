# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import os

import torch
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


def ddp_setup():
    if torch.cuda.is_available():
        init_process_group(backend='nccl')

def get_local_rank() -> int | None:
    local_rank = os.environ.get('LOCAL_RANK')
    return int(local_rank) if local_rank is not None else None

def get_rank() -> int | None:
    rank = os.environ.get('RANK')
    return int(rank) if rank is not None else None

def get_world_size() -> int:
    if not is_ddp_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def torchrun_is_running() -> bool:
    return (get_local_rank() is not None) and (get_world_size() is not None)


def wrap_model_with_ddp_if_needed(
    monogpu_model: torch.nn.Module
) -> torch.nn.Module | DDP:
    if torchrun_is_running():
        gpu_id = get_local_rank()

        # 1. Transfer the model weights to the memory of the assigned GPU.
        monogpu_model = monogpu_model.to(gpu_id)
        # 2. Ensure that the batch normalization layers is parallel-friendly
        monogpu_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            monogpu_model
        )
        # 3. Wrap the model
        model = DDP(monogpu_model, device_ids=[gpu_id])
        return model
    else:
        return monogpu_model


def is_ddp_available_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


