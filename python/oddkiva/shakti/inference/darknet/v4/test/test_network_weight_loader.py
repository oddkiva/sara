from pathlib import Path

from oddkiva.shakti.inference.darknet.v4 import NetworkWeightLoader


THIS_FILE = str(__file__)
SARA_SOURCE_DIR_PATH = Path(THIS_FILE[:THIS_FILE.find('sara') + len('sara')])
SARA_DATA_DIR_PATH = SARA_SOURCE_DIR_PATH / 'data'
YOLO_V4_DIR_PATH = SARA_DATA_DIR_PATH / 'trained_models' / 'yolov4'

YOLO_V4_CFG_PATH = YOLO_V4_DIR_PATH / 'yolov4.cfg'
YOLO_V4_WEIGHT_PATH = YOLO_V4_DIR_PATH / 'yolov4.weights'

assert SARA_DATA_DIR_PATH.exists()
assert YOLO_V4_CFG_PATH.exists()
assert YOLO_V4_WEIGHT_PATH.exists()


def test_yolov4_weight_loader():
    weight_loader = NetworkWeightLoader(YOLO_V4_WEIGHT_PATH)
    print(weight_loader._weights.shape)
