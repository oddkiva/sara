# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from loguru import logger

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader

import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva.brahma.torch.datasets.coco.dataloader import collate_fn
from oddkiva.brahma.torch.object_detection.common.data_transforms import (
    ToNormalizedCXCYWHBoxes
)
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    dn_detr.contrastive_denoising_group_generator import (
        ContrastiveDenoisingGroupGenerator
    )


def get_coco_batch_sample():
    logger.info(f"Instantiating COCO dataset...")
    transform = v2.Compose([
        v2.RandomIoUCrop(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Resize((640, 640)),
        v2.SanitizeBoundingBoxes(),
        ToNormalizedCXCYWHBoxes(),  # IMPORTANT!!!
    ])
    coco_ds = coco.COCOObjectDetectionDataset(
        train_or_val='val',
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

    return sample


def test_contrastive_denoising_group_generator():
    class_count = 80
    box_noise_relative_scale = 1.0
    dn_group_gen = ContrastiveDenoisingGroupGenerator(
        class_count,
        box_noise_relative_scale=box_noise_relative_scale
    )

    images, boxes, labels = get_coco_batch_sample()
    # VERY IMPORTANT
    # --------------
    #
    # - The boxes must be in CXCYWH format.
    # - The box coordinates must be **normalized** in [0, 1]
    #
    # otherwise the denoising group generator can't generate correct boxes.
    batch_size = len(images)

    assert images.shape == (batch_size, 3, 640, 640)
    assert len(boxes) == batch_size
    assert len(labels) == batch_size
    for n in range(batch_size):
        assert len(boxes[n]) == len(labels[n])

    torch.manual_seed(0)
    dn_groups = dn_group_gen.forward(300, boxes, labels)

    B = max([len(b) for b in boxes])
    G = dn_group_gen.box_count // B
    group_size = 2 * B

    assert dn_groups.group_count == G
    assert dn_groups.labels is not None
    assert dn_groups.labels.shape == (batch_size, B * G * 2)
    assert dn_groups.geometries is not None
    assert dn_groups.geometries.shape == (batch_size, B * G * 2, 4)

    # Let's confirm our understanding of how `dn_positive_ixs` is constructed.
    dn_pos_ixs = dn_groups.positive_indices
    assert dn_pos_ixs is not None
    assert len(dn_pos_ixs) == batch_size

    # Check the indexing of the replicated positive (noised ground-truth)
    # boxes.
    #
    # We are now checking that the partition correctly groups the positive
    # boxes by training sample.
    for n, dn_pos_ixs_n in enumerate(dn_pos_ixs):
        # The positive boxes are the replicated ground-truth boxes and they are
        # replicated `G` times.
        #
        # The replicated group   `0` have indices from   0       to 2*B*1-1.
        # The replicated group   `1` have indices from 2*B       to 2*B*2-1.
        # ...
        # The replicated group `G-1` have indices from 2*B*(G-1) to 2*B*G-1.
        num_boxes_n = len(boxes[n])
        pos_ixs_n = torch.cat([
            torch.arange(group_size * g, group_size * g + num_boxes_n)
            for g in range(G)
        ], dim = -1)
        assert torch.equal(pos_ixs_n, dn_pos_ixs_n)

    # Check the geometries of replicated boxes.
    dn_geometries = dn_groups.geometries
    for n in range(1):#batch_size):
        logger.debug(f'Examining geometries for training sample {n}...')
        num_boxes_n = len(boxes[n])

        print(f'boxes[{n}].format\n', boxes[n].format)

        for g in range(G):
            logger.debug(f'Examining for geometries of group ({n}, {g})...')

            whwh = torch.tensor(boxes[n].canvas_size[::-1]).tile(2)[None, ...]
            dn_geoms_n = dn_geometries[n, group_size*g:group_size*(g+1), :]

            boxes_n = boxes[n] * whwh
            pos = (dn_geoms_n[:B] * whwh)[:num_boxes_n]
            neg = (dn_geoms_n[B:] * whwh)[:num_boxes_n]

            print(f'boxes[{n}]\n', boxes_n)
            print(f'pos[{n}, {g}]\n', torch.trunc(pos))
            print(f'neg[{n}, {g}]\n', torch.trunc(neg))
            print(f'pos err[{n}, {g}]\n',
                  torch.dist(pos, boxes_n) / torch.norm(boxes_n))
            print(f'neg err[{n}, {g}]\n',
                  torch.dist(neg, boxes_n) / torch.norm(boxes_n))


    ## Each denoising group g in [0, G[ can be partitioned into two sub-groups:
    ## - the positive groups.
    ## - the negative groups are assigned to label ID `class_count`, which
    ##   corresponds to the 'no-object' category.
    #dn_box_labels = dn_groups.box_labels
    #p_alter = dn_group_gen.box_label_alter_prob * 0.5
    #for n in range(batch_size):
    #    logger.debug(f'Examining for training sample {n}...')
    #    num_boxes_n = len(boxes[n])

    #    # Inspect the labels.
    #    for g in range(G):
    #        logger.debug(f'Examining group {g}...')

    #        pos_labels_altered = dn_box_labels[
    #            n,
    #            group_size*g:group_size*g + num_boxes_n
    #        ]
    #        pos_labels = labels[n]

    #        indices_where_equal = torch.where(pos_labels_altered == pos_labels)[0]

    #        freq_alter = abs(len(indices_where_equal) - num_boxes_n) / num_boxes_n
    #        if num_boxes_n > 10:
    #            print(pos_labels_altered)
    #            print(pos_labels)
    #            print(indices_where_equal)
    #            print(num_boxes_n)
    #            assert abs(freq_alter - p_alter) < 0.2



    # TODO: verify the properties of the contrastive denoising group.
