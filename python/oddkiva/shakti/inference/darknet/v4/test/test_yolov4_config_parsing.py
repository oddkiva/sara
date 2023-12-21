from pathlib import Path

import oddkiva.shakti.inference.darknet as darknet


THIS_FILE = str(__file__)
SARA_SOURCE_DIR_PATH = Path(THIS_FILE[:THIS_FILE.find('sara') + len('sara')])
SARA_MODEL_DIR_PATH = SARA_SOURCE_DIR_PATH / 'trained_models'
YOLO_V4_TINY_DIR_PATH = SARA_MODEL_DIR_PATH / 'yolov4-tiny'

YOLO_V4_TINY_CFG_PATH = YOLO_V4_TINY_DIR_PATH / 'yolov4-tiny.cfg'
YOLO_V4_TINY_WEIGHT_PATH = YOLO_V4_TINY_DIR_PATH / 'yolov4-tiny.weights'

assert SARA_MODEL_DIR_PATH.exists()
assert YOLO_V4_TINY_CFG_PATH.exists()
assert YOLO_V4_TINY_WEIGHT_PATH.exists()


def test_yolo_v4_tiny_cfg():
    yolo_cfg = darknet.Config()
    yolo_cfg.read(YOLO_V4_TINY_CFG_PATH)
    assert yolo_cfg._model is not None

    yolo_net = darknet.Network(yolo_cfg)
