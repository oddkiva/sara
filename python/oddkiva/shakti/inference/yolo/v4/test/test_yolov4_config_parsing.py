from pathlib import Path

from configparser import ConfigParser

from oddkiva.shakti.inference.yolo.darknet_config_parser import (
    DarknetConfigParser
)


THIS_FILE = str(__file__)
SARA_SOURCE_DIR_PATH = Path(THIS_FILE[:THIS_FILE.find('sara') + len('sara')])
SARA_DATA_DIR_PATH = SARA_SOURCE_DIR_PATH / 'data'
YOLO_V4_TINY_DIR_PATH = SARA_DATA_DIR_PATH / 'trained_models'

YOLO_V4_TINY_CFG_PATH = YOLO_V4_TINY_DIR_PATH / 'yolov4-tiny.cfg'

assert SARA_DATA_DIR_PATH.exists()
assert YOLO_V4_TINY_CFG_PATH.exists()


def test_yolo_v4_tiny_conversion():
    parser = DarknetConfigParser()
    parser.read(YOLO_V4_TINY_CFG_PATH)

    print(f'\nmetadata =\n{parser._metadata}')
    print(f'\nmodel')
    for layer in parser._model:
        print(layer)
