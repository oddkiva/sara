from copy import deepcopy

import math

import torch
import torch.nn as nn


def is_parallelized(model) -> bool:
    return type(model) in (torch.nn.parallel.DataParallel,
                           torch.nn.parallel.DistributedDataParallel)

def deparallelize(model) -> nn.Module:
    return model.module if is_parallelized(model) else model


class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self,
                 model: nn.Module,
                 decay: float = 0.9999,
                 warmups: int = 2000):
        super().__init__()

        self.module = deepcopy(deparallelize(model)).eval()

        self.decay = decay
        self.warmups = warmups

        # The current number of EMA updates
        self.updates = 0

        # decay exponential ramp (to help early epochs)
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / warmups))

        # Don't store the gradients, this will save a lot of (GPU) memory.
        for p in self.module.parameters():
            p.requires_grad_(False)


    def update(self, model: nn.Module):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay_fn(self.updates)
            model_state_dict = deparallelize(model).state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * model_state_dict[k].detach()

    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        return self

    def state_dict(self):
        return dict(module=self.module.state_dict(), updates=self.updates)

    def load_state_dict(self, state: dict, strict: bool = True):
        self.module.load_state_dict(state['module'], strict=strict)
        if 'updates' in state:
            self.updates = state['updates']

    def extra_repr(self) -> str:
        return f'decay={self.decay}, warmups={self.warmups}'

