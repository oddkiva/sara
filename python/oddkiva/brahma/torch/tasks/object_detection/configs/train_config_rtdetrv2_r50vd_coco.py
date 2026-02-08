# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import os
from pathlib import Path
from loguru import logger

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter

# Datasets
import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva import DATA_DIR_PATH
# Parallelization
from oddkiva.brahma.torch.parallel.ddp import torchrun_is_running
# Data Transforms
from oddkiva.brahma.torch.datasets.coco.dataloader import (
    RTDETRImageCollateFunction,
    collate_fn
)
from oddkiva.brahma.torch.object_detection.common.data_transforms import (
    FromRgb8ToRgb32f,
    RandomIoUCrop,
    ToNormalizedCXCYWHBoxes
)
from oddkiva.brahma.torch.object_detection.common.mosaic import (
    Mosaic
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
    def make_model() -> RTDETRv2:
        config = RTDETRConfig()
        model = RTDETRv2(config)
        return model


class TrainValTestDatasetConfig:
    Dataset = coco.COCOObjectDetectionDataset
    # NOTE:
    #
    # I can't get more than 5 training samples in a batch on a NVIDIA Titan X
    # (Pascal) 12GB GPU.
    #
    # Unless I freeze:
    # - the 1st convolutional blocks of the PResNet-50 backbone
    # - the batch norm layers of the PResNet-50 backbone,
    # I can get up to 8 samples per batch.
    train_batch_size: int = \
        5   if torch.cuda.is_available() else \
        16  # on my MacBook
    num_workers: int = \
        5 if torch.cuda.is_available() else \
        8 # on my Macbook

    val_batch_size: int = 32

    train_transform: v2.Transform = v2.Compose([
        Mosaic(
            output_size=320,
            rotation_range=10,  # degrees
            translation_range=(0.1, 0.1),
            scaling_range=(0.5, 1.5),
            # NOTE
            #
            # In the first epochs, I deemed it necessary to feed a larger
            # number of bounding boxes per training images, and therefore use
            # the mosaic data transform 80% of the time.
            #
            # Now that we have kept training the model for about 20 cumulated
            # epochs (because of interruptions) like that, I want to see
            # whether by using the mosaic data transform only 50% of the time
            # will improve the detection performance and the object
            # classification.
            #
            # The reason is that while the mosaic data transform provides a
            # larger number of bounding boxes, the cost we pay is that we lose
            # a lot of the richer and finer textural information that
            # characterizes every object of interest because of the pixel
            # sampling.
            #
            # A visual monitoring of the training seems to show that we are
            # able to detect a lot more smaller objects. I observe that every
            # 1000 iterations, on the same test videos the detection of
            # pedestrians get better and better.
            #
            # At some point we really need to do less mosaic data transform so
            # the backbone learns to better represent images and therefore will
            # aid in the classification score.
            #
            # probability=0.8,
            probability=0.5,
            fill_value=0,
            use_cache=False,
            max_cached_images=50,
            random_pop=True
        ),
        v2.RandomPhotometricDistort(p=0.5),
        v2.RandomZoomOut(fill=0, p=0.5),
        RandomIoUCrop(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Resize((640, 640)),
        v2.SanitizeBoundingBoxes(),
        # Sanitize before the box normalization please.
        ToNormalizedCXCYWHBoxes(),
        FromRgb8ToRgb32f(),
    ])

    val_transform: v2.Transform = v2.Compose([
        v2.Resize((640, 640)),
        ToNormalizedCXCYWHBoxes(),
        FromRgb8ToRgb32f(),
    ])

    @staticmethod
    def make_train_dataloader(ds: Dataset) -> DataLoader:
        if torchrun_is_running():
            return DataLoader(
                dataset=ds,
                batch_size=TrainValTestDatasetConfig.train_batch_size,
                collate_fn=RTDETRImageCollateFunction(),
                # The following options are for parallel data training
                sampler=DistributedSampler(ds, shuffle=True),
                num_workers=TrainValTestDatasetConfig.num_workers,
                pin_memory=True
            )
        else:
            return DataLoader(
                dataset=ds,
                shuffle=True,
                batch_size=TrainValTestDatasetConfig.train_batch_size,
                collate_fn=RTDETRImageCollateFunction(),
                pin_memory=torch.cuda.is_available()
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
                pin_memory=True
            )
        else:
            return DataLoader(
                dataset=ds,
                shuffle=False,
                batch_size=TrainValTestDatasetConfig.val_batch_size,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available()
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

    ema_decay: float = 0.9999
    ema_warmup_steps: int = 2000
    # NOTE: because I am starting to learn from scratch.
    activate_ema: bool = False

    gradient_norm_max: float = 0.1


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
