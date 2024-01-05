import logging

import torch

logging.basicConfig(level=logging.DEBUG)


DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
logging.info(f"Default device selected as: {DEFAULT_DEVICE}")
