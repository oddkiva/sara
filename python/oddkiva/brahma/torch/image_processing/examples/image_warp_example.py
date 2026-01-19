# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from pathlib import Path

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

import torch
import torchvision.transforms.v2 as v2

import oddkiva.brahma.torch.image_processing.warp as W
from oddkiva.brahma.torch import DEFAULT_DEVICE


def rotation(theta):
    # fmt: off
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [            0,              0, 1]])
    # fmt: on


THIS_FILE = __file__
SARA_SOURCE_DIR_PATH = Path(THIS_FILE[: THIS_FILE.find("sara") + len("sara")])
SARA_DATA_DIR_PATH = SARA_SOURCE_DIR_PATH / "data"
DOG_IMAGE_PATH = SARA_DATA_DIR_PATH / "dog.jpg"
assert DOG_IMAGE_PATH.exists()

# Image format converters.
to_float_chw = v2.Compose(
    [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
)
to_uint8_hwc = v2.Compose(
    [v2.ToDtype(torch.uint8, scale=True), v2.ToPILImage()]
)


# Image input
image = to_float_chw(Image.open(DOG_IMAGE_PATH)).to(DEFAULT_DEVICE)
image = image[None, :]

# Geometric transform input.
R = torch.Tensor(rotation(np.pi / 6))

# Differential geometry block
H = W.Homography()
H.homography.data = R
H = H.to(DEFAULT_DEVICE)

image_warped = H.forward(image)
image_warped_hwc = to_uint8_hwc(image_warped[0])

plt.imshow(image_warped_hwc)
plt.show()
