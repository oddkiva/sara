# Copyright (C) 2025 David Ok <david.ok8@gmail.com>
#
# --------------------------------------
# Visualizing the graph with Tensorboard
# --------------------------------------
# 
# The user experience in Tensorboard can be very frustrating with the less
# "mainstream" web browsers:
# 
# - On Linux, prefer inspecting the graph with the Chromium browser, fonts are
#   rendered very poorly on Firefox and LibreWolf.
# 
#   ```bash
#   $ sudo apt install -y chromium-browser
#   $ snap install chromium-browser
#   ```
# 
# - On MacOS, likewise prefer inspecting the graph with Safari.

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from oddkiva.brahma.torch.backbone.resnet.vanilla import ResNet50Variant


model = ResNet50Variant()
image = torch.rand(1, 3, 320, 320)

writer = SummaryWriter('runs/model_visualization')
writer.add_graph(model, image)
writer.flush()
writer.close()
