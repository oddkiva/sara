import torch

from oddkiva.brahma.torch.datasets.reid.models.reid_resnet import (
    ReidDescriptor50
)


class LogConfig:
    summary_out_dir = 'train/IUSTPersonReID'
    summary_write_interval = 1


class TrainConfig:
    batch_size = 32


class ReID_Resnet50_256(LogConfig, TrainConfig):
    reid_dim = 256
    image_size = (160, 64)
    Model = ReidDescriptor50

    # Add distributed training configs
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # rank = int(os.environ.get("RANK", 0))

    @staticmethod
    def make() -> torch.nn.Module:
        return ReID_Resnet50_256.Model(ReID_Resnet50_256.reid_dim)
