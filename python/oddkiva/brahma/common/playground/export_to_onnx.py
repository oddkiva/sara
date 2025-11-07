import torch

from oddkiva.brahma.torch.backbone.resnet50 import ResNet50Variant

model = ResNet50Variant()

torch.onnx.export(model, image, 'resnet50_variant.onnx')
