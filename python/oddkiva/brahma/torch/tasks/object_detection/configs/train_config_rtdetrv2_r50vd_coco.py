# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import os
from pathlib import Path
from loguru import logger

import torch
import torchvision.transforms.v2 as v2
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# Datasets
import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva import DATA_DIR_PATH
# Parallelization
from oddkiva.brahma.torch.parallel.ddp import (
    torchrun_is_running,
    wrap_model_with_ddp_if_needed
)
# Data Transforms
from oddkiva.brahma.torch.object_detection.common.data_transforms import (
    ToNormalizedCXCYWHBoxes,
    ToNormalizedFloat32
)
from oddkiva.brahma.torch.datasets.coco.dataloader import (
    RTDETRImageCollateFunction,
    collate_fn
)
# Models
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.config import RTDETRConfig
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.model import RTDETRv2
# Utilities
from oddkiva.brahma.torch.utils.logging import format_msg


class ModelConfig:

    @staticmethod
    def make_model() -> torch.nn.Module | DDP:
        config = RTDETRConfig()
        model = RTDETRv2(config)
        return wrap_model_with_ddp_if_needed(model)


class TrainValTestDatasetConfig:
    Dataset = coco.COCOObjectDetectionDataset
    train_batch_size: int = 4
    val_batch_size: int = 32
    num_workers: int = 4

    train_transform: v2.Transform = v2.Compose([
        v2.RandomIoUCrop(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Resize((640, 640)),
        v2.SanitizeBoundingBoxes(),
        # Sanitize before the box normalization please.
        ToNormalizedCXCYWHBoxes(),
        ToNormalizedFloat32(),
    ])

    val_transform: v2.Transform = v2.Compose([
        v2.Resize((640, 640)),
        ToNormalizedCXCYWHBoxes(),
        ToNormalizedFloat32(),
    ])

    @staticmethod
    def make_train_dataloader(ds: Dataset) -> DataLoader:
        if torchrun_is_running():
            return DataLoader(
                dataset=ds,
                batch_size=TrainValTestDatasetConfig.train_batch_size,
                collate_fn=RTDETRImageCollateFunction(),
                # The following options are for parallel data training
                shuffle=False,
                sampler=DistributedSampler(ds),
                num_workers=TrainValTestDatasetConfig.num_workers,
            )
        else:
            return DataLoader(
                dataset=ds,
                shuffle=True,
                batch_size=TrainValTestDatasetConfig.train_batch_size,
                collate_fn=RTDETRImageCollateFunction(),
                num_workers=TrainValTestDatasetConfig.num_workers
            )

    @staticmethod
    def make_val_dataloader(ds: Dataset) -> DataLoader:
        if torchrun_is_running():
            return DataLoader(
                dataset=ds,
                batch_size=TrainValTestDatasetConfig.val_batch_size,
                collate_fn=collate_fn,
                # The following options are for parallel data training
                shuffle=False,
                sampler=DistributedSampler(ds),
                num_workers=TrainValTestDatasetConfig.num_workers,
            )
        else:
            return DataLoader(
                dataset=ds,
                shuffle=False,
                batch_size=TrainValTestDatasetConfig.val_batch_size,
                collate_fn=collate_fn,
                num_workers=TrainValTestDatasetConfig.num_workers,
            )

    @staticmethod
    def make_datasets() -> tuple[Dataset, Dataset, Dataset | None]:
        logger.info(format_msg(f"Instantiating COCO train dataset..."))
        train_ds = TrainValTestDatasetConfig.Dataset(
            train_or_val='train',
            transform=TrainValTestDatasetConfig.train_transform
        )

        logger.info(format_msg(f"Instantiating COCO val dataset..."))
        val_ds = TrainValTestDatasetConfig.Dataset(
            train_or_val='val',
            transform=TrainValTestDatasetConfig.val_transform
        )

        logger.info(format_msg(
            f"[IGNORING] Instantiating COCO test dataset..."
        ))
        test_ds = None

        return train_ds, val_ds, test_ds


class OptimizationConfig:
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0001


class SummaryWriterConfig:
    out_dir: Path = (DATA_DIR_PATH / 'trained_models' /
                     'rtdetrv2_r50' / 'train' / 'coco' / 'logs')
    write_interval: int = 10

    @staticmethod
    def make_summary_writer() -> SummaryWriter:
        return SummaryWriter(SummaryWriterConfig.out_dir)


class TrainTestPipelineConfig(ModelConfig,
                              TrainValTestDatasetConfig,
                              OptimizationConfig,
                              SummaryWriterConfig):

    trained_model_out_dir = (DATA_DIR_PATH / 'trained_models' /
                             'rtdetrv2_r50' / 'train' / 'coco' / 'ckpts')

    @staticmethod
    def out_model_filepath(epoch: int, step: int | None = None) -> Path:
        if not TrainTestPipelineConfig.trained_model_out_dir.exists():
            os.makedirs(TrainTestPipelineConfig.trained_model_out_dir)
        if step is None:
            return (TrainTestPipelineConfig.trained_model_out_dir /
                    f'ckpt_epoch_{epoch}.pth')
        else:
            return (TrainTestPipelineConfig.trained_model_out_dir /
                    f'ckpt_epoch_{epoch}_step_{step}.pth')
