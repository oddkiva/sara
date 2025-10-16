from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.transforms.v2 as v2

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.datasets.classification_dataset_abc import (
    ClassificationDatasetABC
)
from oddkiva.brahma.torch.datasets.reid.triplet_dataset import TripletDataset
from oddkiva.brahma.torch.datasets.reid.eth123 import ETH123
from oddkiva.brahma.torch.tasks.reid.configs.reid_resnet import (
    ReidDescriptor50
)


class ModelConfig:
    Model: type[torch.nn.Module] = ReidDescriptor50
    reid_dim: int = 256

    # Add distributed training configs
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # rank = int(os.environ.get("RANK", 0))

    @staticmethod
    def make_model() -> torch.nn.Module:
        return ModelConfig.Model(ModelConfig.reid_dim)


class TrainValTestDatasetConfig:
    Dataset = ETH123
    dataset_dir_path: Path = DATA_DIR_PATH / 'reid' / 'dataset_ETHZ'
    image_size: tuple[int, int] = (160, 64)
    batch_size: int = 32

    transforms: v2.Transform = v2.Compose([
        v2.Resize(image_size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])

    @staticmethod
    def make_datasets() -> tuple[ClassificationDatasetABC,
                                 ClassificationDatasetABC,
                                 ClassificationDatasetABC]:
        train_dataset = TrainValTestDatasetConfig.Dataset(
            TrainValTestDatasetConfig.dataset_dir_path,
            transform=TrainValTestDatasetConfig.transforms,
        )
        val_dataset = train_dataset
        test_dataset = train_dataset

        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def make_triplet_dataset(ds: ClassificationDatasetABC) -> DataLoader:
        tds = TripletDataset(ds)
        dl = DataLoader(tds, TrainValTestDatasetConfig.batch_size)
        return dl


class OptimizationConfig:
    learning_rate = 1e-3


class SummaryWriterConfig:
    out_dir: Path = Path('train/dataset_ETHZ')
    write_interval: int = 1

    @staticmethod
    def make_summary_writer() -> SummaryWriter:
        return SummaryWriter(SummaryWriterConfig.out_dir)


class TrainTestPipelineConfig(ModelConfig,
                              TrainValTestDatasetConfig,
                              OptimizationConfig,
                              SummaryWriterConfig):
    pass
