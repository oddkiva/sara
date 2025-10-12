from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
if torch.torch_version.TorchVersion(torch.__version__) < (2, 6, 0):
    from torchvision.io.image import read_image as decode_image
else:
    from torchvision.io.image import decode_image

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.datasets.reid.eth123 import ETH123
from oddkiva.brahma.torch.datasets.reid.models.configs import (
    ReID_Resnet50_256 as ModelConfig
)


# Dataset
ROOT_PATH = DATA_DIR_PATH / 'reid' / 'dataset_ETHZ'
# Data transform
DATA_TRANSFORM = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(ModelConfig.image_size, antialias=True)
])
ETH123_DS = ETH123(ROOT_PATH, transform=DATA_TRANSFORM)


def read_image(image_path: Path) -> torch.Tensor:
    image = decode_image(str(image_path))
    return DATA_TRANSFORM(image)


def dist(a, b):
    return torch.sum((a - b) ** 2, dim=1)


def main():
    # THE DATASET
    train_dataset = ETH123_DS

    # THE TRAINED MODEL
    #
    reid_model = ModelConfig.make()
    # Load the model weights.
    reid_model_weights = torch.load('/home/david/GitLab/oddkiva/sara/eth123_resnet50_0.pt')
    reid_model.load_state_dict(reid_model_weights)
    # Turn off the backpropagation.
    reid_model.eval()

    class_a = train_dataset._image_paths_per_class[0]
    class_b = train_dataset._image_paths_per_class[2]

    with torch.no_grad():
        Xa = read_image(class_a[0])
        da = reid_model(Xa[None, :])

        max_d_ap = -1.
        for i in range(1, len(class_a)):
            Xp = read_image(class_a[i])
            dp = reid_model(Xp[None, :])
            d_ap = dist(da, dp).item()
            max_d_ap = max(max_d_ap, d_ap)

        min_d_an = 1e7
        for j in range(1, len(class_b)):
            Xn = read_image(class_b[j])
            dn = reid_model(Xn[None, :])
            d_an = dist(da, dn).item()
            min_d_an = min(min_d_an, d_an)

        res = ", ".join([f"max_d(a, p)={max_d_ap}", f"min_d(a, n)={min_d_an}"])
        print(res)


if __name__ == "__main__":
    main()
