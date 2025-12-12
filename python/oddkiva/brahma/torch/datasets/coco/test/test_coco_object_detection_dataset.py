# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from oddkiva.brahma.torch.datasets.coco import COCOObjectDetectionDataset


def test_coco_dataset():
    coco = COCOObjectDetectionDataset(train_or_val='train')

    img, bboxes, labels = coco[0]

    assert img.shape == (3, 447, 640)
    assert len(bboxes) == len(labels)
    assert (labels < 80).all()
