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
SARA_YOLOV4_MODEL_DIR_PATH = SARA_TRAINED_MODEL_DIR_PATH / 'yolov4-tiny'

YOLO_V4_COREML_PATH = SARA_YOLOV4_MODEL_DIR_PATH / 'yolov4-tiny.mlpackage'
YOLO_V4_COCO_CLASSES_PATH = SARA_YOLOV4_MODEL_DIR_PATH / 'classes.txt'
assert YOLO_V4_COREML_PATH.exists()
YOLO_V4_CFG_PATH = SARA_YOLOV4_MODEL_DIR_PATH / 'yolov4-tiny.cfg'
assert YOLO_V4_CFG_PATH.exists()

DOG_IMAGE_PATH = SARA_DATA_DIR_PATH / 'dog.jpg'
assert DOG_IMAGE_PATH.exists()


Box = namedtuple('Box', ['x', 'y', 'w', 'h', 'p_object', 'class_id', 'p_class'])


def get_yolo_boxes(yolo_out: np.ndarray, yolo_layers: dict['str': Any],
                   objectness_thres,
                   image_ori_sizes, yolo_input_sizes):
    mask = yolo_layers['mask']
    anchors = yolo_layers['anchors']
    _, B, _, H, W = yolo_out.shape

    out = yolo_out
    rel_x = out[:, :, 0]
    rel_y = out[:, :, 1]
    log_w = out[:, :, 2]
    log_h = out[:, :, 3]
    p_objectness = out[:, :, 4]
    p_classes = out[:, :, 5:]

    yi, xi = np.meshgrid(range(H), range(W), indexing='ij')

    w_prior = [anchors[mask[b]][0] for b in range(B)]
    h_prior = [anchors[mask[b]][1] for b in range(B)]

    sx = image_ori_sizes[1] / yolo_input_sizes[0]
    sy = image_ori_sizes[0] / yolo_input_sizes[1]

    x = (rel_x + xi) / W * image_ori_sizes[1]
    y = (rel_y + yi) / H * image_ori_sizes[0]
    w = log_w
    h = log_h
    for b in range(B):
        w[:, :, b] = np.exp(log_w)[:, :, b] * w_prior[b] * sx
        h[:, :, b] = np.exp(log_h)[:, :, b] * h_prior[b] * sy

    p_class_idx = np.argmax(p_classes, axis=2)

    # Get the 4D indices 
    object_ids = np.nonzero(p_objectness > objectness_thres)
    x = x[object_ids]
    y = y[object_ids]
    w = w[object_ids]
    h = h[object_ids]
    p_objectness = p_objectness[object_ids]
    class_ids = p_class_idx[object_ids]
    ixs = (object_ids[0], object_ids[1], class_ids, object_ids[2],
           object_ids[3])
    p_classes = p_classes[ixs]

    boxes = np.stack((x, y, w, h, p_objectness, class_ids,
                      p_classes)).transpose().tolist()
    boxes = [Box(*b) for b in boxes] 
    return boxes

def nms(boxes_ndarr: [Box], iou_thres=0.4):
    def compare(x: Box, y: Box):
        return x.p_object > y.p_object
    boxes_sorted = sorted(boxes_ndarr, cmp=compare)
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

image_ori = Image.open(DOG_IMAGE_PATH)
image_ori_sizes = np.asarray(image_ori).shape[:2]

image_resized = image_ori.resize(yolo_input_sizes,
                                 resample=Image.Resampling.LANCZOS)
yolo_outs = yolo_model.predict({'image': image_resized})
yolo_outs = [yolo_outs[f'yolo_{i}'] for i in range(len(yolo_layers))]


yolo_boxes = get_yolo_boxes(yolo_outs[0], yolo_layers[0], 0.4, image_ori_sizes,
                            yolo_input_sizes)

import IPython; IPython.embed()
