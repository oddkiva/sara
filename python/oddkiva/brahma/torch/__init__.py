# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import torch


DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Default device selected as: {DEFAULT_DEVICE}")
