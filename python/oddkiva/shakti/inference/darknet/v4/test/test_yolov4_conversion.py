from pathlib import Path
import logging

from PIL import Image

import numpy as np

import torch

# import coremltools as ct

import oddkiva.sara as sara
import oddkiva.shakti.inference.darknet as darknet


logging.basicConfig(level=logging.DEBUG)


THIS_FILE = str(__file__)
SARA_SOURCE_DIR_PATH = Path(THIS_FILE[:THIS_FILE.find('sara') + len('sara')])
SARA_MODEL_DIR_PATH = SARA_SOURCE_DIR_PATH / 'trained_models'
SARA_DATA_DIR_PATH = SARA_SOURCE_DIR_PATH / 'data'
YOLO_V4_DIR_PATH = SARA_MODEL_DIR_PATH / 'yolov4'

YOLO_V4_CFG_PATH = YOLO_V4_DIR_PATH / 'yolov4.cfg'
YOLO_V4_WEIGHT_PATH = YOLO_V4_DIR_PATH / 'yolov4.weights'

DOG_IMAGE_PATH = SARA_DATA_DIR_PATH / 'dog.jpg'

assert SARA_MODEL_DIR_PATH.exists()
assert YOLO_V4_CFG_PATH.exists()
assert YOLO_V4_WEIGHT_PATH.exists()
assert DOG_IMAGE_PATH.exists()


def read_image(path: Path, yolo_net: darknet.Network):
    image = Image.open(path)
    image = np.asarray(image).astype(np.float32) / 255
    image = image.transpose((2, 0, 1))

    _, c, h, w = yolo_net.input_shape()
    image_resized = np.zeros((c, h, w), dtype=np.float32)
    sara.resize(image, image_resized)

    image_tensor = torch.from_numpy(image_resized[np.newaxis, :])
    return image_tensor


def test_yolo_v4_coreml_conversion():
    yolo_cfg = darknet.Config()
    yolo_cfg.read(YOLO_V4_CFG_PATH)
    assert yolo_cfg._model is not None

    # TODO: compare with the weights loaded in C++.
    # There is an implementation error
    layer_idx = 156
    yolo_net = darknet.Network(yolo_cfg, up_to_layer=layer_idx)
    yolo_net.load_convolutional_weights(YOLO_V4_WEIGHT_PATH);
    yolo_net.eval()

    # in_tensor = read_image(DOG_IMAGE_PATH, yolo_net)

    # with torch.inference_mode():
    #     traced_model = torch.jit.trace(yolo_net, in_tensor)
    #     outs = traced_model(in_tensor)

    #     ct_outs = [ct.TensorType(name=f'yolo_{i}') for i, _ in enumerate(outs)]

    #     model = ct.convert(
    #         traced_model,
    #         inputs=[ct.TensorType(shape=in_tensor.shape)],
    #         outputs=ct_outs,
    #         debug=True
    #     )

    #     model.save('/Users/oddkiva/Desktop/yolo-v4.mlpackage')
