from pathlib import Path

from PIL import Image

import numpy as np

import torch

import oddkiva.shakti.inference.darknet as darknet


THIS_FILE = str(__file__)
SARA_SOURCE_DIR_PATH = Path(THIS_FILE[:THIS_FILE.find('sara') + len('sara')])
SARA_MODEL_DIR_PATH = SARA_SOURCE_DIR_PATH / 'trained_models'
SARA_DATA_DIR_PATH = SARA_SOURCE_DIR_PATH / 'data'
YOLO_V4_TINY_DIR_PATH = SARA_MODEL_DIR_PATH / 'yolov4-tiny'

YOLO_V4_TINY_CFG_PATH = YOLO_V4_TINY_DIR_PATH / 'yolov4-tiny.cfg'
YOLO_V4_TINY_WEIGHT_PATH = YOLO_V4_TINY_DIR_PATH / 'yolov4-tiny.weights'

DOG_IMAGE_PATH = SARA_DATA_DIR_PATH / 'dog.jpg'

assert SARA_MODEL_DIR_PATH.exists()
assert YOLO_V4_TINY_CFG_PATH.exists()
assert YOLO_V4_TINY_WEIGHT_PATH.exists()
assert DOG_IMAGE_PATH.exists()


def test_yolo_v4_tiny_cfg():
    yolo_cfg = darknet.Config()
    yolo_cfg.read(YOLO_V4_TINY_CFG_PATH)
    assert yolo_cfg._model is not None

    yolo_net = darknet.Network(yolo_cfg)
    yolo_net.load_convolutional_weights(YOLO_V4_TINY_WEIGHT_PATH);

    image = np.asarray(Image.open(DOG_IMAGE_PATH)).astype(np.float32) / 255
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]

    print(image.shape)

    boxes = yolo_net.forward(torch.from_numpy(image))
    import IPython; IPython.embed()
