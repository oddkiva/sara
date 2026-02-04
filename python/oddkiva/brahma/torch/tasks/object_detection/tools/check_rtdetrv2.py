import sys
from pathlib import Path

from loguru import logger

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize

import oddkiva.sara as sara
import oddkiva.sara.graphics.image_draw as image_draw

from oddkiva import DATA_DIR_PATH
from oddkiva.sara.dataset.colors import generate_label_colors
from oddkiva.brahma.torch.backbone.repvgg import RepVggBlock
from oddkiva.brahma.torch.utils.freeze import freeze_batch_norm
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.config import RTDETRConfig
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.model import RTDETRv2


def optimize_repvgg_layer_for_inference(m: nn.Module):
    if isinstance(m, RepVggBlock):
        logger.info(f'Found RepVGGBlock {m}')
        m.deploy_for_inference()
    else:
        for child_tree_name, child_tree in m.named_children():
            logger.info(f'Exploring {child_tree_name}: {child_tree}')
            optimize_repvgg_layer_for_inference(child_tree)


class ModelConfig:
    CKPT_DIRPATH = (DATA_DIR_PATH / 'trained_models' / 'rtdetrv2_r50' /
                    'train' / 'coco' / 'ckpts')
    CKPT_RESUME_DIRPATH = (DATA_DIR_PATH / 'trained_models' / 'rtdetrv2_r50' /
                           'train' / 'coco' / 'ckpts-resume')
    LABELS_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                       'labels.txt')

    CKPT_DIRPATH.exists()
    CKPT_RESUME_DIRPATH.exists()
    LABELS_FILEPATH.exists()

    W_INFER = 640
    H_INFER = 640
    CONFIDENCE_THRESHOLD = 0.5

    RUN_ON_CPU = False
    LOAD_RESUME_CKPT = False

    RESUME_ITER = 10
    EPOCH = 0
    STEPS = 8000


    @staticmethod
    def load() -> tuple[nn.Module, list[str], torch.device]:

        # This is by design so that we can keep training with the GPU...
        if ModelConfig.RUN_ON_CPU:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:1')

        if ModelConfig.LOAD_RESUME_CKPT:
            filename = '{}-ckpt_epoch_{}_step_{}.pth'.format(
                ModelConfig.RESUME_ITER,
                ModelConfig.EPOCH,
                ModelConfig.STEPS
            )
            CKPT_FP = ModelConfig.CKPT_RESUME_DIRPATH / filename
        else:
            if ModelConfig.STEPS is None:
                CKPT_FP = (
                    ModelConfig.CKPT_DIRPATH /
                    f'ckpt_epoch_{ModelConfig.EPOCH}.pth'
                )
            else:
                CKPT_FP = (
                    ModelConfig.CKPT_DIRPATH /
                    f'ckpt_epoch_{ModelConfig.EPOCH}_step_{ModelConfig.STEPS}.pth'
                )
        assert CKPT_FP.exists()

        # THE MODEL
        config = RTDETRConfig()
        config.setup_for_inference(True)
        model = RTDETRv2(config).to(device)

        # LOAD THE MODEL
        ckpt = torch.load(CKPT_FP, weights_only=True, map_location=device)
        model.load_state_dict(ckpt)

        model = freeze_batch_norm(model)
        optimize_repvgg_layer_for_inference(model)
        model = model.eval()
        # RUN ON THE GPU PLEASE.
        assert type(model) is RTDETRv2

        label_names = [
            l.strip()
            for l in open(ModelConfig.LABELS_FILEPATH, 'r').readlines()
        ]

        return model, label_names, device


def detect_objects(model: nn.Module, rgb_image: np.ndarray, device:
                   torch.device):
    h_ori, w_ori = rgb_image.shape[:2]

    x = torch.from_numpy(rgb_image).to(device=device)
    x = x.permute(2, 0, 1)[None, ...]
    x = resize(x, [ModelConfig.H_INFER, ModelConfig.W_INFER])
    x = x.to(dtype=torch.float32) / 255

    with torch.no_grad():
        object_boxes, object_class_logits, _ = model(x)
        assert object_boxes.requires_grad is False
        assert object_class_logits.requires_grad is False

        boxes = object_boxes[-1, 0]
        probs = F.softmax(object_class_logits[-1, 0], dim=-1)

        N = probs.shape[0]
        labels = torch.argmax(probs, dim=-1)
        confidences = probs[torch.arange(N), labels]

    xc = boxes[:, 0]
    yc = boxes[:, 1]
    widths = boxes[:, 2]
    heights = boxes[:, 3]
    lefts = xc - 0.5 * widths
    tops = yc - 0.5 * heights
    # Rescale
    lefts, widths = (lefts * w_ori, widths * w_ori)
    tops, heights = (tops * h_ori, heights * h_ori)

    # Back to CPU
    lefts = lefts.cpu().numpy()
    tops = tops.cpu().numpy()
    widths = widths.cpu().numpy()
    heights = heights.cpu().numpy()
    labels = labels.cpu().numpy()
    confidences = confidences.cpu().numpy()

    return (lefts, tops, widths, heights, labels, confidences)


def user_main():
    video_file = sys.argv[1]
    assert Path(video_file).exists()
    video_stream = sara.VideoStream()
    video_stream.open(video_file, True)
    h, w, _ = video_stream.sizes()

    model, label_names, device = ModelConfig.load()

    sara.create_window(w, h)
    sara.set_antialiasing(True)

    video_frame = np.empty(video_stream.sizes(), dtype=np.uint8)
    display_frame = np.empty(video_stream.sizes(), dtype=np.uint8)
    video_frame_index = - 1
    video_frame_skip_count = 2

    label_colors = generate_label_colors(len(label_names))

    while video_stream.read(video_frame):
        video_frame_index += 1
        if video_frame_index % (video_frame_skip_count + 1) != 0:
            continue;

        with sara.Timer("ObjectDetection"):
            ls, ts, ws, hs, labels, confs = detect_objects(
                model, video_frame, device
            )
        with sara.Timer("Display"):
            np.copyto(display_frame, video_frame)

            print('frame', video_frame_index)
            for (l, t, w, h, label, conf) in zip(ls, ts, ws, hs,
                                                 labels, confs):
                if conf < ModelConfig.CONFIDENCE_THRESHOLD:
                    continue

                # Draw the object box
                color = label_colors[label]
                xy = (int(l + 0.5), int(t + 0.5))
                wh = (int(w + 0.5), int(h + 0.5))
                image_draw.draw_rect(display_frame, xy, wh, color, 2.)

                # Draw the label
                p = (int(l + 0.5 + 5), int(t + 0.5 - 10))
                text = f'{label_names[label]} {conf:0.2f}'
                font_size = 12
                bold = True

                print(f'[{text}] ({l}, {t}, {w}, {h})')

                image_draw.draw_text(display_frame, p, text, color,
                                     font_size, 0, False, bold, False)

            sara.draw_image(display_frame)


if __name__ == '__main__':
    sara.run_graphics(user_main)
