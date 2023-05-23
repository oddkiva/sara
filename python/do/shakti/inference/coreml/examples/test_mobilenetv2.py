import urllib
from packaging import version
from platform import python_version

from PIL import Image

import numpy as np
from IPython.display import display

import torch
import torchvision

import coremltools as ct


PYTHON_VERSION = version.parse(python_version())
if PYTHON_VERSION.minor > 10:
    raise RuntimeError(
        "This Python version is currently not supported. "
        "See discussion on GitHub in: {}".format(
            "https://github.com/apple/coremltools/issues/1730"
        ))

model = ct.models.MLModel("mobilenet_v2.mlpackage")
spec = model.get_spec()
image_type = spec.description.input[0].type.imageType
w = image_type.width
h = image_type.height


image = Image.open(
    "/Users/oddkiva/GitLab/DO-CV/sara/data/dog.jpg")
image = image.resize((w, h), Image.LANCZOS)
pred = model.predict({'rgbImage': image})
print(pred['classLabel'])
