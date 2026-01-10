# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from loguru import logger

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2

# Data, dataset and dataloader.
import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva.brahma.torch.datasets.coco.dataloader import (
    RTDETRImageCollateFunction
)
# Data augmentation.
from oddkiva.brahma.torch.object_detection.common.data_transforms import (
    ToNormalizedCXCYWHBoxes,
    ToNormalizedFloat32
)
# The model.
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.config import RTDETRConfig
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.model import RTDETRv2
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    dn_detr.contrastive_denoising_group_generator import (
        ContrastiveDenoisingGroupGenerator
    )
# The loss.
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.loss_function import RTDETRHungarianLoss
# GPU acceleration.
from oddkiva.brahma.torch import DEFAULT_DEVICE
# Gradient checks.
from oddkiva.brahma.torch.utils.gradient_tracking import (
    collect_gradients,
    hook_backward,
    hook_forward,
    track_all_layer_gradients
)


def get_coco_val_dl():
    logger.info(f"Instantiating COCO dataset...")
    transform = v2.Compose([
        v2.RandomIoUCrop(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Resize((640, 640)),
        v2.SanitizeBoundingBoxes(),
        # Sanitize before the box normalization please.
        ToNormalizedCXCYWHBoxes(),
        ToNormalizedFloat32(),
    ])
    coco_ds = coco.COCOObjectDetectionDataset(
        train_or_val='val',
        transform=transform
    )

    logger.info(f"Instantiating COCO dataloader...")
    coco_dl = DataLoader(
        dataset=coco_ds,
        batch_size=4,
        shuffle=False,
        collate_fn=RTDETRImageCollateFunction()
    )

    return coco_dl


def get_rtdetrv2_model():
    config = RTDETRConfig()
    model = RTDETRv2(config)
    return model


def test_rtdetrv2_backpropagation_from_anchors():
    gpu0 = DEFAULT_DEVICE

    coco_val_dl = get_coco_val_dl()

    logger.info(f"Getting first batch sample from COCO dataloader...")
    coco_it = iter(coco_val_dl)

    logger.info(f"Instantiating RT-DETR v2 model...")
    rtdetrv2 = get_rtdetrv2_model()
    rtdetrv2 = rtdetrv2.to(gpu0)

    logger.info(
        'Instantianting AdamW optimizer with default hardcoded parameters'
    )
    optimizer = torch.optim.AdamW(rtdetrv2.parameters(),
                                  lr=1e-4,
                                  betas=(0.9, 0.999),
                                  weight_decay=1e-4)
    optimizer.zero_grad()


    logger.info(f"Tracking RT-DETR v2 gradients...")
    layers, grads = track_all_layer_gradients(rtdetrv2, hook_forward, hook_backward)


    # Obtain the first training sample.
    img, tgt_boxes, tgt_labels = next(coco_it)
    img = img.to(gpu0)
    tgt_boxes = [b.to(gpu0) for b in tgt_boxes]
    tgt_labels = [l.to(gpu0) for l in tgt_labels]
    tgt_count = sum([len(l) for l in tgt_labels])
    assert (0 <= img).all() and (img <= 1).all()
    assert tgt_count > 0

    # Feed the input to the object detection network.
    x = img
    targets = {
        'boxes': tgt_boxes,
        'labels': tgt_labels
    }
    box_geoms, box_class_logits, other_train_outputs = rtdetrv2.forward(
        x, targets
    )

    # Calculate the losses.
    weight_dict = {
        'vf': 1.0,
        'box': 1.0
    }
    alpha = 0.2
    gamma = 2.0
    num_classes = 80
    loss_fn = RTDETRHungarianLoss(weight_dict,
                                  alpha=alpha,
                                  gamma=gamma,
                                  num_classes=num_classes)

    # Calculate the loss only for the predicted anchor boxes.
    (anchor_geometry_logits,
     anchor_class_logits) = other_train_outputs['top_k_anchor_boxes']
    anchor_geometries = F.sigmoid(anchor_geometry_logits)

    # The Hungarian loss.
    #
    # 1. Calculate the matching for the anchor boxes.
    matching_a = loss_fn.matcher.forward(
        anchor_class_logits, anchor_geometries,
        tgt_labels, tgt_boxes
    )
    # 2. Calculate the loss value for anchor boxes based on the matching.
    loss_dict = loss_fn.compute_loss_dict(anchor_geometries,
                                          anchor_class_logits,
                                          tgt_boxes, tgt_labels,
                                          matching_a, tgt_count)
    loss = sum([loss_dict[k].sum() for k in loss_dict])
    loss.backward()

    # Update RT-DETR v2 parameters with AdamW.
    optimizer.step()

    # Check the parameters that has changed and those that didn't.
    layer_ixs, grad_values = collect_gradients(grads)

    # Check that we don't update the deformable transformer decoder at all.
    for layer_idx in layer_ixs:
        layer_name = grads[layer_idx][0]
        grad_norm = torch.norm(grad_values[layer_idx])
        logger.debug(f'{layer_name} gradient norm:{grad_norm}')

        assert layer_name.startswith('decoder') is False


def test_rtdetrv2_backpropagation_from_dn_groups():
    gpu0 = DEFAULT_DEVICE

    coco_val_dl = get_coco_val_dl()

    logger.info(f"Getting first batch sample from COCO dataloader...")
    coco_it = iter(coco_val_dl)

    logger.info(f"Instantiating RT-DETR v2 model...")
    rtdetrv2 = get_rtdetrv2_model()
    rtdetrv2 = rtdetrv2.to(gpu0)

    logger.info(
        'Instantianting AdamW optimizer with default hardcoded parameters'
    )
    optimizer = torch.optim.AdamW(rtdetrv2.parameters(),
                                  lr=1e-4,
                                  betas=(0.9, 0.999),
                                  weight_decay=1e-4)
    optimizer.zero_grad()


    logger.info(f"Tracking RT-DETR v2 gradients...")
    layers, grads = track_all_layer_gradients(rtdetrv2, hook_forward, hook_backward)


    # Obtain the first training sample.
    logger.info(f"Obtaining the first training batch...")
    img, tgt_boxes, tgt_labels = next(coco_it)
    img = img.to(gpu0)
    tgt_boxes = [b.to(gpu0) for b in tgt_boxes]
    tgt_labels = [l.to(gpu0) for l in tgt_labels]
    tgt_count = sum([len(l) for l in tgt_labels])
    assert (0 <= img).all() and (img <= 1).all()
    assert tgt_count > 0

    # Feed the input to the object detection network.
    logger.info(f"Feeding the first training batch to RT-DETR v2...")
    x = img
    targets = {
        'boxes': tgt_boxes,
        'labels': tgt_labels
    }
    box_geoms, box_class_logits, aux_train_outs = rtdetrv2.forward(
        x, targets
    )

    # Calculate the losses.
    logger.info(f"Calculating the Hungarian loss for the denoising groups...")
    weight_dict = {
        'vf': 1.0,
        'box': 1.0
    }
    alpha = 0.2
    gamma = 2.0
    num_classes = 80
    loss_fn = RTDETRHungarianLoss(weight_dict,
                                  alpha=alpha,
                                  gamma=gamma,
                                  num_classes=num_classes)

    # Calculate the loss only for the predicted anchor boxes.
    (dn_geometries, dn_class_logits) = aux_train_outs['dn_boxes']
    dn_groups = aux_train_outs['dn_groups']
    assert type(dn_groups) is ContrastiveDenoisingGroupGenerator.Output

    # The Hungarian loss.
    #
    # 1. Calculate the matching for the DN-groups of boxes.
    logger.info('[Hungarian Loss] 1. Matching with the Hungarian algorithm...')
    matching_dn = dn_groups.populate_matching(tgt_labels)

    tgt_boxes_dn = [
        tgt_boxes_n[tixs_n]
        for (tgt_boxes_n, (_, tixs_n)) in zip(tgt_boxes, matching_dn)
    ]
    tgt_labels_dn = [
        tgt_labels_n[tixs_n]
        for (tgt_labels_n, (_, tixs_n)) in zip(tgt_labels, matching_dn)
    ]

    # 2. Calculate the loss value for DN groups of boxes based on the matching.
    # iterations = dn_geometries.shape[0]
    # for i in range(iterations):
    i = -1

    logger.info('[Hungarian Loss] 2. Calculating the composite loss function...')
    loss_dict = loss_fn.compute_loss_dict(dn_geometries[i], dn_class_logits[i],
                                          tgt_boxes_dn, tgt_labels_dn,
                                          matching_dn, tgt_count)
    loss = sum([loss_dict[k].sum() for k in loss_dict])
    loss.backward()

    # Check the parameters that has changed and those that didn't.
    layer_ixs, grad_values = collect_gradients(grads)

    # Check that we don't update the deformable transformer decoder at all.
    for layer_idx in layer_ixs:
        layer_name = grads[layer_idx][0]
        grad_norm = torch.norm(grad_values[layer_idx])
        logger.debug(f'{layer_name} gradient norm:{grad_norm}')


def test_rtdetrv2_backpropagation_from_final_queries():
    gpu0 = DEFAULT_DEVICE

    coco_val_dl = get_coco_val_dl()

    logger.info(f"Getting first batch sample from COCO dataloader...")
    coco_it = iter(coco_val_dl)

    logger.info(f"Instantiating RT-DETR v2 model...")
    rtdetrv2 = get_rtdetrv2_model()
    rtdetrv2 = rtdetrv2.to(gpu0)

    logger.info(
        'Instantianting AdamW optimizer with default hardcoded parameters'
    )
    optimizer = torch.optim.AdamW(rtdetrv2.parameters(),
                                  lr=1e-4,
                                  betas=(0.9, 0.999),
                                  weight_decay=1e-4)
    optimizer.zero_grad()


    logger.info(f"Tracking RT-DETR v2 gradients...")
    layers, grads = track_all_layer_gradients(rtdetrv2, hook_forward, hook_backward)


    # Obtain the first training sample.
    logger.info(f"Obtaining the first training batch...")
    img, tgt_boxes, tgt_labels = next(coco_it)
    img = img.to(gpu0)
    tgt_boxes = [b.to(gpu0) for b in tgt_boxes]
    tgt_labels = [l.to(gpu0) for l in tgt_labels]
    tgt_count = sum([len(l) for l in tgt_labels])
    assert (0 <= img).all() and (img <= 1).all()
    assert tgt_count > 0

    # Feed the input to the object detection network.
    logger.info(f"Feeding the first training batch to RT-DETR v2...")
    x = img
    targets = {
        'boxes': tgt_boxes,
        'labels': tgt_labels
    }
    box_geoms, box_class_logits, _ = rtdetrv2.forward(
        x, targets
    )

    # Calculate the losses.
    logger.info(f"Calculating the Hungarian loss for the final boxes...")
    weight_dict = {
        'vf': 1.0,
        'box': 1.0
    }
    alpha = 0.2
    gamma = 2.0
    num_classes = 80
    loss_fn = RTDETRHungarianLoss(weight_dict,
                                  alpha=alpha,
                                  gamma=gamma,
                                  num_classes=num_classes)

    # The Hungarian loss.
    # for box_logits_i, box_geoms_i in zip(box_class_logits, box_geometries):
    i = 0
    box_logits_i = box_class_logits[i]
    box_geoms_i = box_geoms[i]
    # 1. Calculate the matching for the anchor boxes.
    matching_f = loss_fn.matcher.forward(
        box_logits_i, box_geoms_i,
        tgt_labels, tgt_boxes
    )
    # 2. Calculate the loss value for anchor boxes based on the matching.
    loss_dict = loss_fn.compute_loss_dict(box_geoms_i, box_logits_i,
                                          tgt_boxes, tgt_labels,
                                          matching_f, tgt_count)
    loss = sum([loss_dict[k].sum() for k in loss_dict])
    loss.backward()

    # Update RT-DETR v2 parameters with AdamW.
    optimizer.step()

    # Check the parameters that has changed and those that didn't.
    layer_ixs, grad_values = collect_gradients(grads)

    # Check that we don't update the deformable transformer decoder at all.
    for layer_idx in layer_ixs:
        layer_name = grads[layer_idx][0]
        grad_norm = torch.norm(grad_values[layer_idx])
        logger.debug(f'{layer_name} gradient norm:{grad_norm}')

        assert layer_name.startswith('decoder') is False
