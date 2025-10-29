# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torch.nn as nn


def make_box_embedding(num_boxes, embed_dim: int = 256):
    return nn.Embedding(num_boxes, embed_dim)
