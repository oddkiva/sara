from loguru import logger

import torch as T
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader

import oddkiva.brahma.torch.datasets.coco as coco
from oddkiva.brahma.torch.datasets.coco.dataloader import collate_fn
from oddkiva.brahma.torch.object_detection.common.box_ops import (
    fix_ltrb_coords,
    from_cxcywh_to_ltrb_format,
    from_ltrb_to_cxcywh_format
)
from oddkiva.brahma.torch.losses.focal_loss import focal_loss as fl_func
from oddkiva.brahma.torch.object_detection.common.data_transforms import (
    ToNormalizedCXCYWHBoxes
)
from oddkiva.brahma.torch.object_detection.losses.box_matcher import (
    BoxMatcher
)
from oddkiva.brahma.torch.object_detection.losses.varifocal_loss import (
    VarifocalLoss
)


def get_coco_batched_sample():
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


def test_focal_loss():
    images, tgt_boxes, tgt_labels = get_coco_batched_sample()

    N, C, W, H = 16, 3, 640, 640
    top_K = 300
    assert images.shape == (N, C, W, H)

    num_classes = 80
    tgt_count_per_image = [len(b) for b in tgt_boxes]

    # Generate noise for the box geometries and object class probabilities.
    noise_mag = 15  # pixels
    noise_xyxy = T.rand(N, top_K, 4) * noise_mag / W

    noise_prob_mag = 0.2  # [-0.2, +0.2]
    noise_prob = (2 * T.rand(N, top_K, 80) - 1) * noise_prob_mag

    query_xyxy = noise_xyxy
    query_logits = noise_prob
    for n, (tboxes_n, tlabels_n, tcount_n) in enumerate(
        zip(tgt_boxes, tgt_labels, tgt_count_per_image)
    ):
        # Add the ground-truth boxes on the first rows.
        query_xyxy[n, :tcount_n] += from_cxcywh_to_ltrb_format(tboxes_n)

        # Add the ground-truth object class logits.
        query_logits[n, :tcount_n] += F.one_hot(tlabels_n.to(T.int64),
                                                num_classes=num_classes)

    # Fix the query boxes and put back into cxcywh format.
    query_cxcywh = from_ltrb_to_cxcywh_format(fix_ltrb_coords(query_xyxy))
    query_boxes = query_cxcywh

    matcher = BoxMatcher()
    matching = matcher.forward(query_logits, query_cxcywh, tgt_labels, tgt_boxes)

    vfl = VarifocalLoss()

    vfl_val = vfl.forward(query_boxes, query_logits,
                          tgt_boxes, tgt_labels,
                          matching)
