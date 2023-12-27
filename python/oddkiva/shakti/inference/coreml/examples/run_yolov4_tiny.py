from collections import namedtuple
from pathlib import Path
from typing import Any

from PIL import Image

import numpy as np

import coremltools as ct

import oddkiva.shakti.inference.darknet as darknet


THIS_FILE = __file__
SARA_SOURCE_DIR_PATH = Path(THIS_FILE[:THIS_FILE.find('sara') + len('sara')])
SARA_DATA_DIR_PATH = SARA_SOURCE_DIR_PATH / 'data'
SARA_TRAINED_MODEL_DIR_PATH = SARA_SOURCE_DIR_PATH / 'trained_models'
SARA_YOLOV4_MODEL_DIR_PATH = SARA_SOURCE_DIR_PATH / 'trained_models'

YOLO_V4_COREML_PATH = SARA_YOLOV4_MODEL_DIR_PATH / 'yolo-v4.mlpackage'
YOLO_V4_COCO_CLASSES_PATH = SARA_YOLOV4_MODEL_DIR_PATH / 'classes.txt'
assert YOLO_V4_COREML_PATH.exists()
YOLO_V4_CFG_PATH = SARA_YOLOV4_MODEL_DIR_PATH / 'yolov4.cfg'
assert YOLO_V4_CFG_PATH.exists()

DOG_IMAGE_PATH = SARA_DATA_DIR_PATH / 'dog.jpg'
assert DOG_IMAGE_PATH.exists()


Box = namedtuple('x', 'y', 'w', 'h', 'p_object', 'class_id', 'p_class')


def get_yolo_boxes(yolo_out: np.ndarray, yolo_layers: dict['str': Any],
                   objectness_thres,
                   image_ori_sizes, yolo_input_sizes):
    B = len(yolo_layers['masks'])
    N, C, H, W = yolo_out.shape
    out = yolo_out.reshape((N, B, C // B, H, W))
    rel_x = out[:, :, 0]
    rel_y = out[:, :, 1]
    log_w = out[:, :, 2]
    log_h = out[:, :, 3]
    p_objectness = out[:, :, 4]
    p_classes = out[:, :, 5:]

    yi, xi = np.meshgrid(range(H), range(W), indexing='ij')

    mask = yolo_layers['mask']
    anchors = yolo_layers['anchors']
    w_prior = [anchors[2 * mask[b] + 0] for b in range(B)]
    h_prior = [anchors[2 * mask[b] + 1] for b in range(B)]

    sx = yolo_input_sizes[0] / image_ori_sizes[0]
    sy = yolo_input_sizes[1] / image_ori_sizes[1]

    x = (rel_x + xi) / W * image_ori_sizes[0]
    y = (rel_y + yi) / H * image_ori_sizes[1]
    for b in range(B):
        w = np.exp(log_w)[:, :, b] * w_prior[b] * sx
        h = np.exp(log_h)[:, :, b] * h_prior[b] * sy

    p_class_idx = np.argmax(p_classes, axis=2)

    # Get the 5D indices 
    object_ids = np.where(p_objectness > objectness_thres)
    x = x[object_ids]
    y = y[object_ids]
    w = w[object_ids]
    h = h[object_ids]
    p_objectness = p_objectness[object_ids]
    class_ids = p_class_idx[object_ids]
    p_classes = p_classes[object_ids]

    boxes = np.stack((x, y, w, h, p_objectness, class_ids, p_classes))
    boxes = [Box(b) for b in boxes] 
    return boxes

def nms(boxes_ndarr: [Box], iou_thres=0.4):
    boxes_sorted = sorted(boxes_ndarr,
                          cmp=def compare(x, y): x.p_object > y.p_object)
    boxes_filtered = []
    for box in boxes_sorted:
        if not boxes_filtered:
            boxes_filtered.append(box)
            continue
        x1 = np.array([box.x for box in boxes_filtered])
        y1 = np.array([box.y for box in boxes_filtered])
        w = np.array([box.w for box in boxes_filtered])
        h = np.array([box.h for box in boxes_filtered])
        x2 = x1 + w 
        y2 = y1 + h 

        inter_x1 = np.maximum(x1, box.x)
        inter_y1 = np.maximum(y1, box.y)
        inter_x2 = np.minimum(x2, box.x + box.w)
        inter_y2 = np.minimum(y2, box.y + box.h)

        inter = inter_x1 <= inter_x2 and inter_y1 <= inter_y2
        inter_area = \
            (inter_x2 - inter_x2) * (inter_y2 - inter_y1) * \
            inter.astype(np.float32)

        union_area = w * h + box.w * box.h - inter_area

        iou = inter_area / union_area

        ids = np.where(iou > iou_thres)
        if not ids:
            boxes_filtered.append(box)

    return boxes_filtered


yolo_model = ct.models.CompiledMLModel(str(YOLO_V4_COREML_PATH))
yolo_cfg = darknet.Config()
yolo_cfg.read(YOLO_V4_CFG_PATH)
yolo_input_sizes = (yolo_cfg._metadata['width'], yolo_cfg._metadata['height'])
yolo_layers = [layer['yolo'] for layer in yolo_cfg._model
               if 'yolo' in layer.keys()]

image = Image.open(DOG_IMAGE_PATH).resize(yolo_input_sizes,
                                          resample=Image.Resampling.LANCZOS)

yolo_outs = model.predict({'image': image})
