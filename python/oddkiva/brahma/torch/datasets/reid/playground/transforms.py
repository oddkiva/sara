import matplotlib.pyplot as plt

import torch
import torchvision.transforms.v2 as v2

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.datasets.reid.eth123 import ETH123


DS_DIR_PATH = DATA_DIR_PATH / 'reid' / 'dataset_ETHZ'

h = 320
w = 128
image_size=(h, w)
T = v2.Compose([
    v2.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.1, h / w)),
    v2.RandomPhotometricDistort(brightness=(0.5, 1.5), contrast=(0.5, 1.5),
                                saturation=(0.5, 1.5), hue=(0, 0), p=0.5)
    # v2.RandomHorizontalFlip(p = 0.5)
])
ds = ETH123(
    DS_DIR_PATH,
    transform=v2.Compose([
        v2.Resize(image_size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])
)

X, y = ds[0]

n = 50
TXs = [T(X) for _ in range(n)]
row1 = torch.concat([X] + TXs[:n // 2 - 1], 2)
row2 = torch.concat(TXs[n // 2 - 1:-1], 2)
mosaic = torch.concat((row1, row2), 1)
plt.imshow(mosaic.permute(1, 2, 0))
plt.show()
