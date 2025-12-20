# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import pickle
from loguru import logger

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from torchvision.ops import box_convert

import oddkiva.sara as sara
import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva import DATA_DIR_PATH
from oddkiva.sara.dataset.colors import generate_label_colors
from oddkiva.brahma.torch.datasets.coco.dataloader import collate_fn
from oddkiva.brahma.torch.object_detection.common.data_transforms import (
    BoundingBoxes,
    ToNormalizedCXCYWHBoxes
)
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    dn_detr.contrastive_denoising_group_generator import (
        ContrastiveDenoisingGroupGenerator
    )


def get_or_create_coco_dataset(
    force_recreate: bool = False
) -> coco.COCOObjectDetectionDataset:
    coco_fp = DATA_DIR_PATH / 'coco_object_detection_dataset.pkl'
    if force_recreate:
        logger.info("Reserializing the COCO object detection dataset...")
    if coco_fp.exists() and not force_recreate:
        logger.info( f'Loading COCO Dataset object from file: {coco_fp}...')
        with open(coco_fp, 'rb') as f:
            coco_ds = pickle.load(f)
            assert type(coco_ds) is coco.COCOObjectDetectionDataset
    else:
        with sara.Timer("COCO Dataset Generation"):
            logger.info( f'Generating COCO Dataset object from file: {coco_fp}')
            transform = v2.Compose([
                v2.RandomIoUCrop(),
                v2.RandomHorizontalFlip(p=0.5),
                v2.Resize((640, 640)),
                v2.SanitizeBoundingBoxes(),
                ToNormalizedCXCYWHBoxes()
            ])
            coco_ds = coco.COCOObjectDetectionDataset(
                train_or_val='train',
                transform=transform
            )
        with sara.Timer("COCO Dataset Serialization"):
            logger.info( f'Serializing COCO Dataset object to file: {coco_fp}...')
            with open(coco_fp, 'wb') as f:
                pickle.dump(coco_ds, f, protocol=pickle.HIGHEST_PROTOCOL)

    return coco_ds


def get_or_create_coco_batch_sample(
    force_recreate: bool = False
) -> tuple[torch.Tensor, list[BoundingBoxes], list[torch.Tensor]]:
    COCO_BATCH_SAMPLE = DATA_DIR_PATH / 'coco_batch_sample.pkl'
    if COCO_BATCH_SAMPLE.exists() and not force_recreate:
        logger.info(f"Loading COCO batch sample from {COCO_BATCH_SAMPLE}...")
        with open(COCO_BATCH_SAMPLE, 'rb') as f:
            sample = pickle.load(f)
    else:
        logger.info(f"Instantiating COCO dataset...")
        transform = v2.Compose([
            v2.RandomIoUCrop(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Resize((640, 640)),
            v2.SanitizeBoundingBoxes(),
            ToNormalizedCXCYWHBoxes(),  # IMPORTANT!!!
        ])
        coco_ds = coco.COCOObjectDetectionDataset(
            train_or_val='train',
            transform=transform
        )

        logger.info(f"Instantiating COCO dataloader...")
        coco_dl = DataLoader(
            dataset=coco_ds,
            batch_size=16,
            shuffle=False,
            collate_fn=collate_fn
        )

        logger.info(f"Getting first batch sample from COCO dataloader...")
        coco_it = iter(coco_dl)
        sample = next(coco_it)

        with open(COCO_BATCH_SAMPLE, 'wb') as f:
            pickle.dump(sample, f)

    return sample


def from_normalized_cxcywh_to_ltwh(boxes: BoundingBoxes) -> torch.Tensor:
    whwh = torch.tensor(boxes.canvas_size[::-1]).tile(2)[None, ...]
    boxes_rescaled = box_convert(boxes * whwh, 'cxcywh', 'xywh')
    return boxes_rescaled


def get_ltwh_boxes(
    box_geometry_logits: torch.Tensor,
    hw: tuple[int, int]
) -> torch.Tensor:
    boxes = F.sigmoid(box_geometry_logits)
    whwh = torch.tensor(hw[::-1]).tile(2)[None, ...]
    boxes = box_convert(boxes * whwh, 'cxcywh', 'xywh')
    return boxes


def draw_dn_boxes(dn_boxes: torch.Tensor, dn_labels: torch.Tensor,
                  class_count: int, label_colors: np.ndarray, font: QFont):
    # The maximum number of boxes in this batch
    B = 21
    # The group size (positive and negative boxes)
    group_size = B * 2
    for i, box in enumerate(dn_boxes.tolist()):
        x, y, w, h = box

        is_negative = (i % group_size) >= 21
        label = int(dn_labels[i].item())
        if label == class_count:
            continue

        # Dim the color to show it is a noised box.
        if is_negative:
            label_color = (127, 0, 0)
            label_name = f'N {label}'
        else:
            label_color = (label_colors[label] * 0.33).astype(np.int32)
            label_name = f'P {label}'

        sara.draw_rect((x, y), (w, h), label_color, 3)
        sara.draw_boxed_text(x, y, label_name, label_color, font, angle=0.)


def draw_gt_boxes(boxes: BoundingBoxes, labels: torch.Tensor,
                  label_colors: np.ndarray, font: QFont):
    for box, label in zip(boxes.tolist(), labels.tolist()):
        x, y, w, h = box

        label_color = label_colors[label]
        label_name = str(label)

        sara.draw_rect((x, y), (w, h), label_color, 5)
        sara.draw_boxed_text(x, y, label_name, label_color, font, angle=0.)


def user_main():
    class_count = 80
    query_count = 300
    img, boxes, labels = get_or_create_coco_batch_sample()

    # NOTE:
    # The original implementation of the denoising groups sets the relative
    # noise scale as 1.0.
    #
    # This is really large in my opinion. It's very surprising to me that it
    # lead to very good detection performance.
    #
    # Setting it to 0.5 makes more sense to me...
    #
    # Next it remains to see as for how the negative samples are being reused
    # in the training stage.
    noise_relative_scale = 0.5
    dn_gen = ContrastiveDenoisingGroupGenerator(
        class_count,
        box_noise_relative_scale=noise_relative_scale
    )
    dng = dn_gen.forward(query_count, boxes, labels)

    # Display config
    font = sara.make_font()
    label_colors = generate_label_colors(class_count)

    sara.create_window(640, 640)

    n = 0

    for n in range(len(img)):
        sara.draw_image(img[n].permute(1, 2, 0).contiguous().numpy())
        # Show the noised boxes first before the ground-truth boxes for clarity.
        dn_boxes = get_ltwh_boxes(dng.box_geometry_logits[n], boxes[n].canvas_size)
        dn_labels = dng.box_labels[n]
        draw_dn_boxes(dn_boxes, dn_labels, class_count, label_colors, font)

        # Now show the ground-truth boxes.
        boxes_n = from_normalized_cxcywh_to_ltwh(boxes[n])
        labels_n = labels[n]
        draw_gt_boxes(boxes_n, labels_n, label_colors, font)

        while True:
            key_pressed = sara.get_key()
            if key_pressed == Qt.Key.Key_Escape:
                logger.info("Terminating...")
                return
            if key_pressed == Qt.Key.Key_Space:
                logger.info("Next training sample")
                break


sara.run_graphics(user_main)
