# Copyright (C) 2025 David Ok <david.ok8@gmail.com>
#
# From: https://docs.pytorch.org/tutorials/intermediate/visualizing_gradients_tutorial.html

import torch


def hook_forward(module_name: str, grads, hook_backward):
    def hook(module, args, output):
        """Forward pass hook which attaches backward pass hooks to intermediate tensors"""
        if isinstance(output, torch.Tensor):
            if output.requires_grad:
                output.register_hook(hook_backward(module_name, grads))
        elif isinstance(output, tuple):
            for out_i in output:
                if isinstance(out_i, torch.Tensor) and out_i.requires_grad:
                    out_i.register_hook(hook_backward(module_name, grads))
        else:
            raise NotImplementedError()
    return hook


def hook_backward(module_name: str, grads):
    def hook(grad):
        """Backward pass hook which appends gradients"""
        grads.append((module_name, grad))
    return hook


def track_all_layer_gradients(model: torch.nn.Module, hook_forward, hook_backward):
    """Register forward pass hook (which registers a backward hook) to model outputs

    Returns:
        layers:
            a dict with keys as layer/module and values as layer/module names
            e.g. layers[nn.Conv2d] = layer1.0.conv1
        grads:
            a list of tuples with module name and tensor output gradient, e.g.
            grads[0] == (layer1.0.conv1, tensor.Torch(...))
    """
    layers = dict()
    grads = []
    for name, layer in model.named_modules():
        # skip Sequential and/or wrapper modules
        if any(layer.children()) is False:
            layers[layer] = name
            layer.register_forward_hook(hook_forward(name, grads, hook_backward))
    return layers, grads


def collect_gradients(grads):
    layer_idx = []
    avg_grads = []
    for idx, (_, grad) in enumerate(grads):
        if grad is not None:
            avg_grad = grad.abs().mean()
            avg_grads.append(avg_grad)
            # idx is backwards since we appended in backward pass
            layer_idx.append(len(grads) - 1 - idx)
    return layer_idx, avg_grads



# Usage: how to register hooks?
# layers_bn, grads_bn = get_all_layers(model_bn, hook_forward, hook_backward)
# layers_nobn, grads_nobn = get_all_layers(model_nobn, hook_forward, hook_backward)
