from pathlib import Path

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

import torch
import torchvision.transforms.v2 as v2

import oddkiva.brahma.torch.image_processing.warp as W
from oddkiva.brahma.torch import DEFAULT_DEVICE


def rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])


THIS_FILE = __file__
SARA_SOURCE_DIR_PATH = Path(THIS_FILE[:THIS_FILE.find('sara') + len('sara')])
SARA_DATA_DIR_PATH = SARA_SOURCE_DIR_PATH / 'data'
DOG_IMAGE_PATH = SARA_DATA_DIR_PATH / 'dog.jpg'
assert DOG_IMAGE_PATH.exists()

to_float_chw = v2.Compose([v2.ToImage(),
                           v2.ToDtype(torch.float32, scale=True)])
image = to_float_chw(Image.open(DOG_IMAGE_PATH)).to(DEFAULT_DEVICE)

# Geometric transform.
R = torch.Tensor(rotation(np.pi / 6))
Rinv = torch.Tensor(R.T).to(DEFAULT_DEVICE)

# Enumerate all the coordinates.
h, w = image[0].shape
p = W.enumerate_coords(w, h).to(DEFAULT_DEVICE)
pinv = torch.matmul(Rinv, p.float())

# Differentiable geometric computational block
interp = W.BilinearInterpolation2d().to(DEFAULT_DEVICE)

image_warped = torch.zeros_like(image)

for c in range(3):
    values, ixs_flat = interp.forward(image[c], pinv)
    image_warped_plane_c = image_warped[c].flatten()
    image_warped_plane_c[ixs_flat] = values

image_warped_np = torch.permute(image_warped, (1, 2, 0)).cpu().numpy()

plt.imshow(image_warped_np)
plt.show()
