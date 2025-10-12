# I don't like YAML config file, I would rather make use the Python parser.

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.transforms.v2 as v2

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.datasets.classification_dataset_abc import (
    ClassificationDatasetABC
)
from oddkiva.brahma.torch.datasets.reid.triplet_dataset import TripletDataset
from oddkiva.brahma.torch.datasets.reid.iust_person_reid import IUSTPersonReID
from oddkiva.brahma.torch.datasets.reid.configs.reid_resnet import (
    ReidDescriptor50
)


class ModelConfig:
    Model = ReidDescriptor50
    reid_dim = 256

    # Add distributed training configs
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # rank = int(os.environ.get("RANK", 0))

    @staticmethod
    def make_model() -> torch.nn.Module:
        return ModelConfig.Model(ModelConfig.reid_dim)


class TrainValTestDatasetConfig:
    dataset_dir_path = DATA_DIR_PATH / 'reid' / 'IUSTPersonReID'
    dataset_class = IUSTPersonReID
    image_size = (160, 64)
    batch_size = 32

    transforms = v2.Compose([
        v2.Resize(image_size, antialias=True),
        v2.ToDtype(torch.float32, scale=True)
    ])

    @staticmethod
    def make_datasets():
        train_dataset = IUSTPersonReID(
            TrainValTestDatasetConfig.dataset_dir_path,
            transform=TrainValTestDatasetConfig.transforms,
            dataset_type='train'
        )
        val_dataset = IUSTPersonReID(
            TrainValTestDatasetConfig.dataset_dir_path,
            transform=TrainValTestDatasetConfig.transforms,
            dataset_type='test'
        )
        test_dataset = val_dataset

        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def make_triplet_dataset(ds: ClassificationDatasetABC):
        tds = TripletDataset(ds)
        dl = DataLoader(tds, TrainValTestDatasetConfig.batch_size)
        return dl


class LogConfig:
    summary_out_dir = 'train/IUSTPersonReID'
    summary_write_interval = 1

    @staticmethod
    def make_summary_writer() -> SummaryWriter:
        return SummaryWriter(LogConfig.summary_out_dir)


class TrainTestPipelineConfig(ModelConfig,
                              TrainValTestDatasetConfig,
                              LogConfig):

    def __init__(self):
        super(TrainTestPipelineConfig, self).__init__()
        assert self.dataset_dir_path.exists()
