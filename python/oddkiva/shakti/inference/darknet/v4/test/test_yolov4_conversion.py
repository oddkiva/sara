from pathlib import Path
import logging

from PIL import Image

import numpy as np

import torch

import oddkiva.sara as sara
import oddkiva.shakti.inference.darknet as darknet
import oddkiva.shakti.inference.coreml.yolo_v4 as ct


logging.basicConfig(level=logging.DEBUG)


# Sara directories.
THIS_FILE = str(__file__)
SARA_SOURCE_DIR_PATH = Path(THIS_FILE[:THIS_FILE.find('sara') + len('sara')])
SARA_MODEL_DIR_PATH = SARA_SOURCE_DIR_PATH / 'trained_models'
SARA_DATA_DIR_PATH = SARA_SOURCE_DIR_PATH / 'data'

# YOLO v4 trained model.
YOLO_V4_DIR_PATH = SARA_MODEL_DIR_PATH / 'yolov4'
YOLO_V4_CFG_PATH = YOLO_V4_DIR_PATH / 'yolov4.cfg'
YOLO_V4_WEIGHT_PATH = YOLO_V4_DIR_PATH / 'yolov4.weights'
YOLO_V4_CXX_DATA_CHECK_DIR_PATH = YOLO_V4_DIR_PATH / "data_check"

# Image.
DOG_IMAGE_PATH = SARA_DATA_DIR_PATH / 'dog.jpg'

assert SARA_MODEL_DIR_PATH.exists()
assert YOLO_V4_CFG_PATH.exists()
assert YOLO_V4_WEIGHT_PATH.exists()
assert YOLO_V4_CXX_DATA_CHECK_DIR_PATH.exists()
assert DOG_IMAGE_PATH.exists()


def read_tensor(path: Path, dims: int):
    logging.debug(f'Read tensor file = {path}')
    with open(path, 'rb') as fp:
        shape = np.fromfile(fp, dtype=np.int32, count=dims)
        logging.debug(f'Read shape = {shape}')
        x = np.fromfile(fp, dtype=np.float32,
                        count=shape.prod()).reshape(shape)
        x = torch.from_numpy(x)
        return x

def read_image(path: Path, yolo_net: darknet.Network):
    image = Image.open(path)
    image = np.asarray(image).astype(np.float32) / 255
    image = image.transpose((2, 0, 1))

    _, c, h, w = yolo_net.input_shape()
    image_resized = np.zeros((c, h, w), dtype=np.float32)
    sara.resize(image, image_resized)

    image_tensor = torch.from_numpy(image_resized[np.newaxis, :])
    return image_tensor


def conv_weight_paths(id: int):
    return (YOLO_V4_CXX_DATA_CHECK_DIR_PATH / f'conv_weight_{id}.bin',
            YOLO_V4_CXX_DATA_CHECK_DIR_PATH / f'conv_bias_{id}.bin')


def yolo_out_path(id: int):
    return YOLO_V4_CXX_DATA_CHECK_DIR_PATH / f'yolo_inter_{id}.bin'


def read_conv_weights(id: int):
    w_path, b_path = conv_weight_paths(id)
    return (read_tensor(w_path, 4), read_tensor(b_path, 1))

def read_yolo_out_tensor(id: int):
    path = yolo_out_path(id)
    return read_tensor(path, 4)


def test_yolo_v4_weights():
    yolo_cfg = darknet.Config()
    yolo_cfg.read(YOLO_V4_CFG_PATH)
    assert yolo_cfg._model is not None

    # Load all the C++ convolutional weights.
    conv_weights = [read_conv_weights(i) for i in range(110)]

    layer_idx = None
    yolo_net = darknet.Network(yolo_cfg, up_to_layer=layer_idx)
    yolo_net.load_convolutional_weights(YOLO_V4_WEIGHT_PATH);
    yolo_net.eval()

    conv_blocks = [block for block in yolo_net.model
                   if type(block) is darknet.ConvBNA]

    err_max = 5e-6
    for i, conv_bn_a in enumerate(conv_blocks):
        conv = conv_bn_a.layers[0]
        logging.info(f'[{i}] Checking fused convolution {conv}')
        assert type(conv) is torch.nn.Conv2d

        w, b = conv_weights[i]
        assert torch.norm(conv.weight.data - w).item() < err_max
        assert torch.norm(conv.bias.data - b).item() < err_max

def test_yolo_v4_prediction():
    yolo_cfg = darknet.Config()
    yolo_cfg.read(YOLO_V4_CFG_PATH)

    layer_idx = None
    yolo_net = darknet.Network(yolo_cfg, up_to_layer=layer_idx)
    yolo_net.load_convolutional_weights(YOLO_V4_WEIGHT_PATH);
    yolo_net.eval()

    in_tensor = read_image(DOG_IMAGE_PATH, yolo_net)

    ys, _ = yolo_net._forward(in_tensor)
    assert len(ys) == len(yolo_net.model) + 1
    assert torch.equal(ys[0], in_tensor)

    err_max = 5e-4
    for i in range(1, len(yolo_net.model)):
        block = yolo_net.model[i]
        out_tensor_saved = read_yolo_out_tensor(i + 1)
        out_tensor_computed = ys[i + 1]

        if type(block) is darknet.Yolo:
            n, b, c, h, w = out_tensor_computed.shape
            out_tensor_computed = out_tensor_computed.reshape(n, b * c, h, w)

        assert out_tensor_saved.shape == out_tensor_computed.shape

        err = torch.max(torch.abs(out_tensor_computed - out_tensor_saved)).item()
        logging.info(f'[{i}] err = {err} for {block}')
        assert err < err_max


def test_yolo_v4_coreml_conversion():
    yolo_cfg = darknet.Config()
    yolo_cfg.read(YOLO_V4_CFG_PATH)

    layer_idx = None
    yolo_net = darknet.Network(yolo_cfg, up_to_layer=layer_idx)
    yolo_net.load_convolutional_weights(YOLO_V4_WEIGHT_PATH);
    yolo_net.eval()

    in_tensor = read_image(DOG_IMAGE_PATH, yolo_net)

    ct.convert_yolo_v4_to_coreml(yolo_net, in_tensor,
                                 YOLO_V4_DIR_PATH / "yolo-v4.mlpackage")
