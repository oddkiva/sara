import torch
from torch.utils.tensorboard.writer import SummaryWriter

from oddkiva.brahma.torch.backbone.resnet50 import ResNet50Variant


model = ResNet50Variant()
image = torch.rand(1, 3, 320, 320)

writer = SummaryWriter('runs/model_visualization')
writer.add_graph(model, image)
writer.flush()
writer.close()

torch.onnx.export(model, image, 'resnet50_variant.onnx')
