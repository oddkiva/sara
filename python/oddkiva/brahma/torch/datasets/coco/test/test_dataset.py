# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from oddkiva.brahma.torch.dataset.coco import COCOObjectDetectionDataset

def test_coco_dataset():
    coco = COCOObjectDetectionDataset('train')
