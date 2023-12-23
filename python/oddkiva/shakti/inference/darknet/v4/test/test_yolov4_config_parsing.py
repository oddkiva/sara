from pathlib import Path
import logging

from PIL import Image

import numpy as np

import torch

import coremltools as ct

import oddkiva.sara as sara
import oddkiva.shakti.inference.darknet as darknet


logging.basicConfig(level=logging.DEBUG)


THIS_FILE = str(__file__)
SARA_SOURCE_DIR_PATH = Path(THIS_FILE[:THIS_FILE.find('sara') + len('sara')])
SARA_MODEL_DIR_PATH = SARA_SOURCE_DIR_PATH / 'trained_models'
SARA_DATA_DIR_PATH = SARA_SOURCE_DIR_PATH / 'data'
YOLO_V4_TINY_DIR_PATH = SARA_MODEL_DIR_PATH / 'yolov4-tiny'

YOLO_V4_TINY_CFG_PATH = YOLO_V4_TINY_DIR_PATH / 'yolov4-tiny.cfg'
YOLO_V4_TINY_WEIGHT_PATH = YOLO_V4_TINY_DIR_PATH / 'yolov4-tiny.weights'

DOG_IMAGE_PATH = SARA_DATA_DIR_PATH / 'dog.jpg'
YOLO_INTER_OUT_DIR_PATH = Path('/Users/oddkiva/Desktop/yolo-intermediate-out')

assert SARA_MODEL_DIR_PATH.exists()
assert YOLO_V4_TINY_CFG_PATH.exists()
assert YOLO_V4_TINY_WEIGHT_PATH.exists()
assert DOG_IMAGE_PATH.exists()


def yolo_out_path(id: int):
    return YOLO_INTER_OUT_DIR_PATH / f'yolo_inter_{id}.bin'

def yolo_out_tensor(id: int):
    yp = yolo_out_path(id)
    with open(yp, 'rb') as fp:
        shape = np.fromfile(fp, dtype=np.int32, count=4)
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


# def test_yolo_v4_tiny_cfg():
#     yolo_cfg = darknet.Config()
#     yolo_cfg.read(YOLO_V4_TINY_CFG_PATH)
#     assert yolo_cfg._model is not None
# 
#     yolo_net = darknet.Network(yolo_cfg)
#     yolo_net.load_convolutional_weights(YOLO_V4_TINY_WEIGHT_PATH);
#     yolo_net.eval()
# 
#     in_tensor = read_image(DOG_IMAGE_PATH, yolo_net)
#     in_tensor_saved = yolo_out_tensor(0)
#     err = torch.norm(in_tensor - in_tensor_saved).item()
#     logging.info(f'input err = {err}')
#     assert err < 1e-12
# 
#     ys, boxes = yolo_net._forward(in_tensor)
#     assert len(ys) == len(yolo_net.model) + 1
#     assert torch.equal(ys[0], in_tensor)
# 
#     for i in range(1, len(yolo_net.model)):
#         block = yolo_net.model[i]
#         out_tensor_saved = yolo_out_tensor(i + 1)
#         out_tensor_computed = ys[i + 1]
# 
#         assert out_tensor_saved.shape == out_tensor_computed.shape
# 
#         err = torch.norm(out_tensor_computed - out_tensor_saved).item()
#         logging.info(f'[{i}] err = {err} for {block}')
#         assert err < 3e-3
# 
# 
#     ids = [31, 38]
#     boxes_true = [yolo_out_tensor(id) for id in ids]
# 
#     for i, b, b_true in zip(ids, boxes, boxes_true):
#         err = torch.norm(b - b_true)
#         logging.info(f'[{i}] err = {err} for {yolo_net.model[i-1]}')
#         assert err < 1e-4


def test_yolo_v4_tiny_coreml_conversion():
    yolo_cfg = darknet.Config()
    yolo_cfg.read(YOLO_V4_TINY_CFG_PATH)
    assert yolo_cfg._model is not None

    layer_idx = 37
    yolo_net = darknet.Network(yolo_cfg, up_to_layer=layer_idx)
    yolo_net.load_convolutional_weights(YOLO_V4_TINY_WEIGHT_PATH);
    yolo_net.eval()

    print(yolo_net.model)

    in_tensor = read_image(DOG_IMAGE_PATH, yolo_net)
    in_tensor_saved = yolo_out_tensor(0)

    with torch.inference_mode():
        traced_model = torch.jit.trace(yolo_net, in_tensor)
        outs = traced_model(in_tensor)

        ct_outs = [ct.TensorType(name=f'yolo_{i}') for i, _ in enumerate(outs)]

        model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=in_tensor.shape)],
            debug=True
        )

        model.save('/Users/oddkiva/Desktop/yolo.mlpackage')

    import IPython; IPython.embed()
