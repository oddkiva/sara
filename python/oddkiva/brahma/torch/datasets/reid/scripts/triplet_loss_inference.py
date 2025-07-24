import platform
from pathlib import Path

import torch
import torch.nn
import torchvision
import torchvision.transforms.v2 as v2
if torch.torch_version.TorchVersion(torch.__version__) < (2, 6, 0):
    from torchvision.io.image import read_image as decode_image
else:
    from torchvision.io.image import decode_image

from oddkiva.brahma.torch.datasets.reid.eth123 import ETH123


class ModelConfig:
    reid_dim = 256
    image_size = (160, 64)
    batch_size = 32
    summary_write_interval = 1

    # Add distributed training configs
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # rank = int(os.environ.get("RANK", 0))


# Dataset
if platform.system() == 'Darwin':
    ROOT_PATH = Path('/Users/oddkiva/Downloads/reid/dataset_ETHZ/')
else:
    ROOT_PATH = Path('/home/david/GitLab/oddkiva/sara/data/reid/dataset_ETHZ/')
# Data transform
DATA_TRANSFORM = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(ModelConfig.image_size, antialias=True)
])
ETH123_DS = ETH123(ROOT_PATH, transform=DATA_TRANSFORM)


class ReidDescriptor50(torch.nn.Module):

    def __init__(self, dim: int = 256):
        super(ReidDescriptor50, self).__init__()
        self.resnet50_backbone = torch.nn.Sequential(
            *list(torchvision.models.resnet50().children())[:-1]
        )
        self.linear = torch.nn.Linear(2048, dim)
        self.softmax = torch.nn.Softmax()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.resnet50_backbone(x)
        y = torch.flatten(y, start_dim=1)
        y = self.linear(y)
        y = self.softmax(y)
        return y


def main():
    # THE DATASET
    train_dataset = ETH123_DS

    # THE TRAINED MODEL
    #
    model_weights = torch.load('/home/david/GitLab/oddkiva/sara/eth123_resnet50_0.pt')
    model = ReidDescriptor50()
    model.load_state_dict(model_weights)
    model.eval()

    class_a = train_dataset._image_paths_per_class[0]
    class_b = train_dataset._image_paths_per_class[2]


    def read_image(image_path: Path) -> torch.Tensor:
        image = decode_image(str(image_path))
        return DATA_TRANSFORM(image)

    def dist(a, b):
        return torch.sum((a - b) ** 2, dim=1)

    Xa = read_image(class_a[0])
    da = model(Xa[None, :])

    max_d_ap = -1.
    for i in range(1, len(class_a)):
        Xp = read_image(class_a[i])
        dp = model(Xp[None, :])
        d_ap = dist(da, dp).item()
        max_d_ap = max(max_d_ap, d_ap)

    min_d_an = 1e7
    for j in range(1, len(class_b)):
        Xn = read_image(class_b[j])
        dn = model(Xn[None, :])
        d_an = dist(da, dn).item()
        min_d_an = min(min_d_an, d_an)

    res = ", ".join([f"max_d(a, p)={max_d_ap}", f"min_d(a, n)={min_d_an}"])
    print(res)


if __name__ == "__main__":
    main()
