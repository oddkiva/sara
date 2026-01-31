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
from oddkiva.brahma.torch.object_detection.common.mosaic import (
    Mosaic
)
from oddkiva.brahma.torch.object_detection.common.data_transforms import (
    RandomIoUCrop,
    ToNormalizedCXCYWHBoxes,
    FromRgb8ToRgb32f
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
    rt_detr.hungarian_loss import (
        HungarianLossReducer,
        RTDETRHungarianLoss
    )
# GPU acceleration.
from oddkiva.brahma.torch import DEFAULT_DEVICE
# Gradient checks.
from oddkiva.brahma.torch.utils.gradient_tracking import (
    collect_gradients,
    hook_backward,
    hook_forward,
    track_all_layer_gradients
)


DEVICE = DEFAULT_DEVICE if DEFAULT_DEVICE != 'mps' else 'cpu'


def get_coco_val_dl():
    logger.info(f"Instantiating COCO dataset...")
    transform = v2.Compose([
        Mosaic(
            output_size=320,
            rotation_range=10,
            translation_range=(0.1, 0.1),
            scaling_range=(0.5, 1.5),
            probability=1.0,
            fill_value=0,
            use_cache=False,
            max_cached_images=50,
            random_pop=True
        ),
        v2.RandomPhotometricDistort(p=0.5),
        v2.RandomZoomOut(fill=0, p=0.5),
        RandomIoUCrop(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Resize((640, 640)),
        v2.SanitizeBoundingBoxes(),
        # Sanitize before the box normalization please.
        ToNormalizedCXCYWHBoxes(),
        FromRgb8ToRgb32f(),
    ])
    coco_ds = coco.COCOObjectDetectionDataset(
        train_or_val='val',
        transform=transform
    )

    logger.info(f"Instantiating COCO dataloader...")
    coco_dl = DataLoader(
        dataset=coco_ds,
        batch_size=4,
        shuffle=True,
        collate_fn=RTDETRImageCollateFunction(),
        pin_memory=True
    )

    return coco_dl


def get_rtdetrv2_model():
    config = RTDETRConfig()
    model = RTDETRv2(config)
    return model


def test_rtdetrv2_backpropagation_from_anchors():
    coco_val_dl = get_coco_val_dl()
    logger.info(f"Getting first batch sample from COCO dataloader...")
    coco_it = iter(coco_val_dl)

    logger.info(f"Instantiating RT-DETR v2 model...")
    rtdetrv2 = get_rtdetrv2_model()
    rtdetrv2 = rtdetrv2.to(DEVICE)

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
    img = img.to(DEVICE)
    tgt_boxes = [b.to(DEVICE) for b in tgt_boxes]
    tgt_labels = [l.to(DEVICE) for l in tgt_labels]
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
    logger.info(f"Calculating the Hungarian loss for the anchor queries...")
    classification_loss_params = {
        'alpha': 0.75,
        'gamma': 2.0
    }
    box_matcher_params = {
        'alpha': 0.25,
        'gamma': 2.0,
        'cost_matrix_weights': {
            'class': 2.0,
            'l1': 5.0,
            'giou': 2.0
        }
    }
    loss_fn = RTDETRHungarianLoss(
        classification_loss_params=classification_loss_params,
        box_matcher_params=box_matcher_params

    )
    # Check the focal loss parameters.
    assert loss_fn.focal_loss.alpha == classification_loss_params['alpha']
    assert loss_fn.focal_loss.gamma == classification_loss_params['gamma']
    # Check the varifocal loss parameters.
    assert loss_fn.varifocal_loss.alpha == classification_loss_params['alpha']
    assert loss_fn.varifocal_loss.gamma == classification_loss_params['gamma']
    # Check the focal loss parameters for the classification cost matrix.
    assert loss_fn.matcher.alpha == box_matcher_params['alpha']
    assert loss_fn.matcher.gamma == box_matcher_params['gamma']
    # Check the weights for the cost matrices.
    cost_matrix_weights = box_matcher_params['cost_matrix_weights']
    assert loss_fn.matcher.w_class == cost_matrix_weights['class']
    assert loss_fn.matcher.w_box_l1 == cost_matrix_weights['l1']
    assert loss_fn.matcher.w_box_giou == cost_matrix_weights['giou']

    # Calculate the loss only for the predicted anchor boxes.
    anchor_boxes, anchor_class_logits = other_train_outputs['anchors']

    # The Hungarian loss.
    #
    # 1. Calculate the matching for the anchor boxes.
    matching_a = loss_fn.matcher.forward(
        anchor_class_logits, anchor_boxes,
        tgt_labels, tgt_boxes
    )
    # 2. Calculate the loss value for anchor boxes based on the matching.
    loss_dict = loss_fn.compute_loss_dict(anchor_boxes, anchor_class_logits,
                                          tgt_boxes, tgt_labels,
                                          matching_a, tgt_count)
    loss = torch.stack([loss_dict[k] for k in loss_dict]).sum()
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

    logger.info(f'Elementary losses:\n{loss_dict}')


def test_rtdetrv2_backpropagation_from_dn_groups():
    coco_val_dl = get_coco_val_dl()

    logger.info(f"Getting first batch sample from COCO dataloader...")
    coco_it = iter(coco_val_dl)

    logger.info(f"Instantiating RT-DETR v2 model...")
    rtdetrv2 = get_rtdetrv2_model()
    rtdetrv2 = rtdetrv2.to(DEVICE)

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
    img = img.to(DEVICE)
    tgt_boxes = [b.to(DEVICE) for b in tgt_boxes]
    tgt_labels = [l.to(DEVICE) for l in tgt_labels]
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
    _, _, aux_train_outs = rtdetrv2.forward(
        x, targets
    )

    # Calculate the losses.
    logger.info(f"Calculating the Hungarian loss for the denoising groups...")
    classification_loss_params = {
        'alpha': 0.75,
        'gamma': 2.0
    }
    box_matcher_params = {
        'alpha': 0.25,
        'gamma': 2.0,
        'cost_matrix_weights': {
            'class': 2.0,
            'l1': 5.0,
            'giou': 2.0
        }
    }
    loss_fn = RTDETRHungarianLoss(
        classification_loss_params=classification_loss_params,
        box_matcher_params=box_matcher_params

    )
    # Check the focal loss parameters.
    assert loss_fn.focal_loss.alpha == classification_loss_params['alpha']
    assert loss_fn.focal_loss.gamma == classification_loss_params['gamma']
    assert loss_fn.focal_loss.alpha == 0.75
    assert loss_fn.focal_loss.gamma == 2.0
    # Check the varifocal loss parameters.
    assert loss_fn.varifocal_loss.alpha == classification_loss_params['alpha']
    assert loss_fn.varifocal_loss.gamma == classification_loss_params['gamma']
    assert loss_fn.varifocal_loss.alpha == 0.75
    assert loss_fn.varifocal_loss.gamma == 2.0
    # Check the focal loss parameters for the classification cost matrix.
    assert loss_fn.matcher.alpha == box_matcher_params['alpha']
    assert loss_fn.matcher.gamma == box_matcher_params['gamma']
    # Check the weights for the cost matrices.
    cost_matrix_weights = box_matcher_params['cost_matrix_weights']
    assert loss_fn.matcher.w_class == cost_matrix_weights['class']
    assert loss_fn.matcher.w_box_l1 == cost_matrix_weights['l1']
    assert loss_fn.matcher.w_box_giou == cost_matrix_weights['giou']
    assert loss_fn.matcher.w_class == 2.0
    assert loss_fn.matcher.w_box_l1 == 5.0
    assert loss_fn.matcher.w_box_giou == 2.0

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
    tgt_count_dn = tgt_count * dn_groups.group_count
    loss_dict = loss_fn.compute_loss_dict(dn_geometries[i], dn_class_logits[i],
                                          tgt_boxes_dn, tgt_labels_dn,
                                          matching_dn, tgt_count_dn)
    logger.debug(f'Elementary losses:\n{loss_dict}')
    loss = torch.stack([loss_dict[k] for k in loss_dict]).sum()
    loss.backward()

    # Check the parameters that has changed and those that didn't.
    layer_ixs, grad_values = collect_gradients(grads)

    # Check that we don't update the deformable transformer decoder at all.
    for layer_idx in layer_ixs:
        layer_name = grads[layer_idx][0]
        grad_norm = torch.norm(grad_values[layer_idx])
        logger.debug(f'{layer_name} gradient norm:{grad_norm}')

    logger.info(f'Elementary losses:\n{loss_dict}')


def test_rtdetrv2_backpropagation_from_final_queries():
    coco_val_dl = get_coco_val_dl()

    logger.info(f"Getting first batch sample from COCO dataloader...")
    coco_it = iter(coco_val_dl)

    logger.info(f"Instantiating RT-DETR v2 model...")
    rtdetrv2 = get_rtdetrv2_model()
    rtdetrv2 = rtdetrv2.to(DEVICE)

    logger.info(
        'Instantianting AdamW optimizer with default hardcoded parameters'
    )
    optimizer = torch.optim.AdamW(rtdetrv2.parameters(),
                                  lr=1e-4,
                                  betas=(0.9, 0.999),
                                  weight_decay=1e-4)
    optimizer.zero_grad()


    logger.info(f"Tracking RT-DETR v2 gradients...")
    _, grads = track_all_layer_gradients(rtdetrv2, hook_forward, hook_backward)


    # Obtain the first training sample.
    logger.info(f"Obtaining the first training batch...")
    img, tgt_boxes, tgt_labels = next(coco_it)
    img = img.to(DEVICE)
    tgt_boxes = [b.to(DEVICE) for b in tgt_boxes]
    tgt_labels = [l.to(DEVICE) for l in tgt_labels]
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
    classification_loss_params = {
        'alpha': 0.75,
        'gamma': 2.0
    }
    box_matcher_params = {
        'alpha': 0.25,
        'gamma': 2.0,
        'cost_matrix_weights': {
            'class': 2.0,
            'l1': 5.0,
            'giou': 2.0
        }
    }
    loss_fn = RTDETRHungarianLoss(
        classification_loss_params=classification_loss_params,
        box_matcher_params=box_matcher_params

    )
    # Check the focal loss parameters.
    assert loss_fn.focal_loss.alpha == classification_loss_params['alpha']
    assert loss_fn.focal_loss.gamma == classification_loss_params['gamma']
    assert loss_fn.focal_loss.alpha == 0.75
    assert loss_fn.focal_loss.gamma == 2.0
    # Check the varifocal loss parameters.
    assert loss_fn.varifocal_loss.alpha == classification_loss_params['alpha']
    assert loss_fn.varifocal_loss.gamma == classification_loss_params['gamma']
    assert loss_fn.varifocal_loss.alpha == 0.75
    assert loss_fn.varifocal_loss.gamma == 2.0
    # Check the focal loss parameters for the classification cost matrix.
    assert loss_fn.matcher.alpha == box_matcher_params['alpha']
    assert loss_fn.matcher.gamma == box_matcher_params['gamma']
    # Check the weights for the cost matrices.
    cost_matrix_weights = box_matcher_params['cost_matrix_weights']
    assert loss_fn.matcher.w_class == cost_matrix_weights['class']
    assert loss_fn.matcher.w_box_l1 == cost_matrix_weights['l1']
    assert loss_fn.matcher.w_box_giou == cost_matrix_weights['giou']
    assert loss_fn.matcher.w_class == 2.0
    assert loss_fn.matcher.w_box_l1 == 5.0
    assert loss_fn.matcher.w_box_giou == 2.0

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
    loss = torch.stack([loss_dict[k] for k in loss_dict]).sum()
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

    logger.info(f'Elementary losses:\n{loss_dict}')


def test_hungarian_loss_api():
    coco_val_dl = get_coco_val_dl()

    logger.info(f"Getting first batch sample from COCO dataloader...")
    coco_it = iter(coco_val_dl)

    logger.info(f"Instantiating RT-DETR v2 model...")
    rtdetrv2 = get_rtdetrv2_model()
    rtdetrv2 = rtdetrv2.to(DEVICE)

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
    img = img.to(DEVICE)
    tgt_boxes = [b.to(DEVICE) for b in tgt_boxes]
    tgt_labels = [l.to(DEVICE) for l in tgt_labels]
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
    box_geoms, box_class_logits, aux_train_outputs = rtdetrv2.forward(
        x, targets
    )

    # Calculate the losses.
    logger.info(f"Calculating the Hungarian loss for the final boxes...")
    classification_loss_params = {
        'alpha': 0.75,
        'gamma': 2.0
    }
    box_matcher_params = {
        'alpha': 0.25,
        'gamma': 2.0,
        'cost_matrix_weights': {
            'class': 2.0,
            'l1': 5.0,
            'giou': 2.0
        }
    }
    loss_fn = RTDETRHungarianLoss(
        classification_loss_params=classification_loss_params,
        box_matcher_params=box_matcher_params

    )
    # Check the focal loss parameters.
    assert loss_fn.focal_loss.alpha == classification_loss_params['alpha']
    assert loss_fn.focal_loss.gamma == classification_loss_params['gamma']
    assert loss_fn.focal_loss.alpha == 0.75
    assert loss_fn.focal_loss.gamma == 2.0
    # Check the varifocal loss parameters.
    assert loss_fn.varifocal_loss.alpha == classification_loss_params['alpha']
    assert loss_fn.varifocal_loss.gamma == classification_loss_params['gamma']
    assert loss_fn.varifocal_loss.alpha == 0.75
    assert loss_fn.varifocal_loss.gamma == 2.0
    # Check the focal loss parameters for the classification cost matrix.
    assert loss_fn.matcher.alpha == box_matcher_params['alpha']
    assert loss_fn.matcher.gamma == box_matcher_params['gamma']
    # Check the weights for the cost matrices.
    cost_matrix_weights = box_matcher_params['cost_matrix_weights']
    assert loss_fn.matcher.w_class == cost_matrix_weights['class']
    assert loss_fn.matcher.w_box_l1 == cost_matrix_weights['l1']
    assert loss_fn.matcher.w_box_giou == cost_matrix_weights['giou']
    assert loss_fn.matcher.w_class == 2.0
    assert loss_fn.matcher.w_box_l1 == 5.0
    assert loss_fn.matcher.w_box_giou == 2.0

    anchor_boxes, anchor_class_logits = aux_train_outputs['anchors']
    dn_boxes, dn_class_logits = aux_train_outputs['dn_boxes']
    dn_groups = aux_train_outputs['dn_groups']

    loss_dict = loss_fn.forward(
        box_geoms, box_class_logits,
        anchor_boxes, anchor_class_logits,
        dn_boxes, dn_class_logits, dn_groups,
        tgt_boxes, tgt_labels
    )

    loss_reducer = HungarianLossReducer({
        'vf': 1.0,
        'l1': 5.0,
        'giou': 2.0
    })
    loss = loss_reducer.forward(loss_dict)
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

    for k in loss_dict:
        logger.info(f'[{k}] loss_dict = {loss_dict[k]}')
