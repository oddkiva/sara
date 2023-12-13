import torch

from do.shakti.inference.yolo.v4.darknet2pytorch import Darknet

YOLO_V4_CFG = 'path/to/cfg/yolov4-416.cfg'
YOLO_V4_WEIGHTS = 'path/to/cfg/yolov4-416.weights'
YOLO_V4_PTH = 'path/to/save/yolov4-pytorch.pth'


# load weights from darknet format
model = darknet2pytorch.Darknet(YOLO_V4_CFG, inference=True)
model.load_weights(YOLO_V4_WEIGHTS)

# save weights to pytorch format
torch.save(model.state_dict(), YOLO_V4_PTH)

# reload weights from pytorch format
model_pt = darknet2pytorch.Darknet(YOLO_V4_CFG, inference=True)
model_pt.load_state_dict(torch.load(YOLO_V4_PTH))
