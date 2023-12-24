from pathlib import Path

from PIL import Image

import coremltools as ct

YOLO_V4_ML_PATH = Path('/Users/oddkiva/Desktop/yolo-v4-tiny.mlpackage/')

THIS_FILE = __file__
SARA_SOURCE_DIR_PATH = Path(THIS_FILE[:THIS_FILE.find('sara') + len('sara')])
SARA_DATA_DIR_PATH = SARA_SOURCE_DIR_PATH / 'data'
DOG_IMAGE_PATH = SARA_DATA_DIR_PATH / 'dog.jpg'

assert YOLO_V4_ML_PATH.exists()
assert DOG_IMAGE_PATH.exists()

model = ct.models.CompiledMLModel(str(YOLO_V4_ML_PATH))

image = Image.open(DOG_IMAGE_PATH).resize(
    (416, 416),
    resample=Image.Resampling.LANCZOS)

yolo_boxes = model.predict({'image': image})
