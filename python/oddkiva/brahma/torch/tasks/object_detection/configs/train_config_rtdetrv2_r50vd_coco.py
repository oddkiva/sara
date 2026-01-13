# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from loguru import logger

from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter

import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva.brahma.torch.parallel.ddp import (
    torchrun_is_running,
    wrap_model_with_ddp_if_needed
)
from oddkiva.brahma.torch.object_detection.common.data_transforms import (
    ToNormalizedCXCYWHBoxes,
    ToNormalizedFloat32
)
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.config import RTDETRConfig
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.model import RTDETRv2


class ModelConfig:

    @staticmethod
    def make_model() -> torch.nn.Module | DDP:
        config = RTDETRConfig()
        model = RTDETRv2(config)
        return wrap_model_with_ddp_if_needed(model)


class TrainValTestDatasetConfig:
    Dataset = coco.COCOObjectDetectionDataset
    batch_size: int = 4
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
        ToNormalizedFloat32(),
    ])

    @staticmethod
    def make_dataloader(
        ds: Dataset,
        batch_size: int
    ) -> DataLoader:
        if torchrun_is_running():
            return DataLoader(
                dataset=ds,
                batch_size=batch_size,
                # The following options are for parallel data training
                shuffle=False,
                sampler=DistributedSampler(ds),
                num_workers=TrainValTestDatasetConfig.num_workers
            )
        else:
            return DataLoader(
                dataset=ds,
                batch_size=batch_size,
                shuffle=True
            )

    @staticmethod
    def make_datasets() -> tuple[DataLoader, DataLoader, DataLoader | None]:
        logger.info(f"Instantiating COCO train dataset...")
        train_ds = TrainValTestDatasetConfig.Dataset(
            train_or_val='train',
            transform=TrainValTestDatasetConfig.train_transform
        )
        train_dl = TrainValTestDatasetConfig.make_dataloader(
            train_ds,
            TrainValTestDatasetConfig.batch_size
        )

        logger.info(f"Instantiating COCO val dataset...")
        val_ds = TrainValTestDatasetConfig.Dataset(
            train_or_val='val',
            transform=TrainValTestDatasetConfig.val_transform
        )
        val_dl = TrainValTestDatasetConfig.make_dataloader(
            val_ds,
            TrainValTestDatasetConfig.batch_size
        )

        logger.info(f"[IGNORING] Instantiating COCO test dataset...")
        test_ds = None

        return train_dl, val_dl, test_ds


class OptimizationConfig:
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0001


class SummaryWriterConfig:
    out_dir: Path = Path('train/coco')
    write_interval: int = 1

    @staticmethod
    def make_summary_writer() -> SummaryWriter:
        return SummaryWriter(SummaryWriterConfig.out_dir)


class TrainTestPipelineConfig(ModelConfig,
                              TrainValTestDatasetConfig,
                              OptimizationConfig,
                              SummaryWriterConfig):
    pass
