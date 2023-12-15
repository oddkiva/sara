import urllib
from packaging import version
from platform import python_version

from PIL import Image

import numpy as np

import torch
import torchvision
from torchvision import transforms

import coremltools as ct


PYTHON_VERSION = version.parse(python_version())
if PYTHON_VERSION.minor > 10:
    raise RuntimeError(
        "This Python version is currently not supported. "
        "See discussion on GitHub in: {}".format(
            "https://github.com/apple/coremltools/issues/1730"
        ))

torch_model = torchvision.models.mobilenet_v2(pretrained=True)
torch_model.eval()

# Get the class names.
label_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
class_labels = urllib.request.urlopen(label_url).read().splitlines()
# Remove the background class.
class_labels = class_labels[1:]
assert len(class_labels) == 1000
for i, label in enumerate(class_labels):
    if isinstance(label, bytes):
        class_labels[i] = label.decode("utf-8")
classifier_config = ct.ClassifierConfig(class_labels)

image = Image.open(
    "/Users/oddkiva/GitLab/DO-CV/sara/data/dog.jpg")
image = image.resize((224, 224), Image.LANCZOS)
to_tensor = transforms.ToTensor()
input_tensor = to_tensor(image)
input_batch = input_tensor.unsqueeze(0)
traced_model = torch.jit.trace(torch_model, input_batch)

input_type = ct.ImageType(
    name="rgbImage",
    shape=input_batch.shape,
    color_layout=ct.colorlayout.RGB,
    bias=[-0.485/0.229, -0.456/0.224, - 0.406/0.225],
    scale=1/(0.226*255.0)
)


model = ct.convert(
    traced_model,
    inputs=[input_type],
    classifier_config=classifier_config,
)

# Rename the input.
model.input_description["rgbImage"] = "Input image to be classified"
model.output_description["classLabel"] = "Most likely image category"
model.author = '"Original Paper: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen'
model.license = "Please see https://github.com/tensorflow/tensorflow for license information, and https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet for the original source of the model."
model.short_description = "Detects the dominant objects present in an image from a set of 1001 categories such as trees, animals, food, vehicles, person etc. The top-1 accuracy from the original publication is 74.7%."
model.version = "2.0"

model.save("mobilenet_v2.mlpackage")
