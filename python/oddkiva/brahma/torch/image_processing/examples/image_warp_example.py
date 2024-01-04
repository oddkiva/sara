from pathlib import Path

from PIL import Image

import pylab

import torch
import torchvision.transforms.v2 as v2

# from oddkiva.brahma.torch.image_processing import rotate, upscale


def rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     np.sin(theta),  np.cos(theta)]])


THIS_FILE = __file__
SARA_SOURCE_DIR_PATH = Path(THIS_FILE[:THIS_FILE.find('sara') + len('sara')])
SARA_DATA_DIR_PATH = SARA_SOURCE_DIR_PATH / 'data'
DOG_IMAGE_PATH = SARA_DATA_DIR_PATH / 'dog.jpg'
assert DOG_IMAGE_PATH.exists()

to_float_chw = v2.Compose([v2.ToImage(),
                           v2.ToDtype(torch.float32, scale=True)])
image = to_float_chw(Image.open(DOG_IMAGE_PATH))


R = rotation(0.5)


# pylab.imshow(image_scaled.astype(np.uint8))
# pylab.show()
