# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNA(nn.Module):

    def __init__(self, in_channels, darknet_params: dict[str, Any], id: int,
                 inference=True):
        super(ConvBNA, self).__init__()
        self.layers = nn.Sequential()

        # Unpack the block parameters from the Darknet configuration.
        batch_normalize = darknet_params['batch_normalize']
        out_channels = darknet_params['filters']
        kernel_size = darknet_params['size']
        stride = darknet_params['stride']
        add_padding = darknet_params['pad']
        activation = darknet_params['activation']
        pad_size = (kernel_size - 1) // 2 if add_padding else 0

        self.batch_normalize = batch_normalize
        self.fuse_conv_bn_layer = inference and batch_normalize

        # Add the convolutional layer
        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding=pad_size,
            bias=True,
            padding_mode='zeros' # Let's be explicit about the padding value
        )
        self.layers.add_module(f'conv_{id}', conv)

        # Add the batch-normalization layer
        if self.batch_normalize and not self.fuse_conv_bn_layer:
            self.layers.add_module(f'batch_norm_{id}', nn.BatchNorm2d(out_channels))
        else:
            self.bn_weights = {
                'scales': None,
                'running_mean': None,
                'running_var': None,
            }

        # Add the activation layer
        if activation == 'leaky':
            activation_fn = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation == 'mish':
            activation_fn = nn.Mish()
        elif activation == 'linear':
            activation_fn = nn.Identity(inplace=True)
        elif activation == 'logistic':
            activation_fn = nn.Sigmoid()
        else:
            raise ValueError(f'No convolutional activation named {activation}')
        self.layers.add_module(f'{activation}_{id}', activation_fn);

    def forward(self, x):
        return self.layers.forward(x)


class MaxPool(nn.Module):

    def __init__(self, kernel_size, stride):
        super(MaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        # self.kernel_size = [kernel_size, kernel_size]
        # from functools import reduce
        # from operator import __add__
        # self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
        #     [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, x):
        # x_padded = self.zero_pad_2d(x)

        # Let's use shortcut variables.
        s = self.stride
        # Get the height and width of the input signal.
        h, w = x.shape[2:]

        # The kernel radius is calculated as
        r = self.kernel_size // 2

        # We calculate the easy part of the padding.
        p_left = r - 1 if self.kernel_size % 2 == 0 else r
        p_top = r - 1 if self.kernel_size % 2 == 0 else r

        # Now moving on the trickiest part of the padding.
        #
        # The max pool layers collects (w // s) x (h // s) samples from the
        # input signal.
        #
        # If we reason in 1D, the samples are located at:
        #   0, s, 2s, 3s, ... , (w // s) * s
        #
        # The input signal is extended spatially so that it contains the
        # following sample points.
        x_last = ((w - 1) // s) * s + r
        y_last = ((h - 1) // s) * s + r
        # Therefore the last two padding are
        p_right = 0 if x_last == w - 1 else x_last - w + 1
        p_bottom = 0 if y_last == h - 1 else y_last - h + 1

        # Apply the padding with negative infinity value.
        pad_size = (p_left, p_right, p_top, p_bottom)
        x_padded = F.pad(x, pad_size, mode='constant', value=-float('inf'))
        # print(f'x_padded = \n{x_padded}')

        # Apply the spatial max-pool function.
        y = F.max_pool2d(x_padded, self.kernel_size, self.stride)

        return y


class RouteSlice(nn.Module):

    def __init__(self, layer: int,
                 groups: int = 1,
                 group_id: Optional[int] = None,
                 id: Optional[int] = None):
        super(RouteSlice, self).__init__()
        self.layer = layer
        self.groups = groups
        self.group_id = group_id
        self.id = id

    def forward(self, x):
        if self.groups == 1:
            return x
        else:
            # Get the number of channels.
            _, C, _, _ = x.shape
            # Split the channels into multiple groups.
            group_size = C // self.groups
            # Get the slice that we want.
            c1 = self.group_id * group_size
            c2 = c1 + group_size
            return x[:, c1:c2, :, :]


class RouteConcat2(nn.Module):

    def __init__(self, layers: list[int], id: Optional[int] = None):
        super(RouteConcat2, self).__init__()
        self.layers = layers
        self.id = id

    def forward(self, x1, x2):
        if len(self.layers) != 2:
            raise RuntimeError(f"This route-concat layer requires 2 inputs")
        return torch.cat((x1, x2), 1)


class RouteConcat4(nn.Module):

    def __init__(self, layers: list[int], id: Optional[int] = None):
        super(RouteConcat4, self).__init__()
        self.layers = layers
        self.id = id

    def forward(self, x1, x2, x3, x4):
        if len(self.layers) != 4:
            raise RuntimeError(f"This route-concat layer requires 4 inputs")
        return torch.cat((x1, x2, x3, x4), 1)


class Shortcut(nn.Module):

    def __init__(self, from_layer: int, activation: str):
        super(Shortcut, self).__init__()
        self.from_layer = from_layer
        if activation == 'linear':
            self.activation_fn = nn.Identity()
        elif activation == 'leaky':
            self.activation_fn = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'relu':
            self.activation_fn = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError(
                f'The followig activation function "{activation}" not implemented!'
            )

    def forward(self, x1, x2):
        x = self.activation_fn(x1 + x2)
        return x


class Upsample(nn.Module):

    def __init__(self, stride: int):
        super(Upsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=stride, mode='nearest')

    def forward(self, x):
        return self.upsample(x)


class Yolo(nn.Module):

    def __init__(self, darknet_params: dict[str, Any]):
        super(Yolo, self).__init__()
        self.masks = darknet_params['mask']
        self.anchors = darknet_params['anchors']
        self.scale_x_y = darknet_params['scale_x_y']
        self.num_classes = darknet_params['classes']

        self.alpha = self.scale_x_y
        self.beta = -0.5 * (self.scale_x_y - 1)

    def forward(self, x):
        num_box_features = 5 + self.num_classes
        assert num_box_features == 85

        # Copy the input.
        y = torch.clone(x)

        # Reshape the 4D tensor as a 5D tensor.
        # We do so by splitting the list of channels by boxes.
        b = len(self.masks)
        n, ci, h, w = x.shape
        co = ci // b
        y = y.reshape((n, b, co, h, w))

        # Box positions
        xs = 0
        ys = 1
        y[:, :, xs] = self.alpha * torch.sigmoid(y[:, :, xs]) + self.beta
        y[:, :, ys] = self.alpha * torch.sigmoid(y[:, :, ys]) + self.beta

        # Box sizes: DO NOTHING since they are already copied
        #
        # ws = 2
        # hs = 3
        # y[:, :, ws] = x[:, :, ws]
        # y[:, :, hs] = x[:, :, hs]

        # P[object] and all P[class|object] probabilities.
        prob_objects = 4
        y[:, :, prob_objects:num_box_features] = \
            torch.sigmoid(y[:, :, prob_objects:num_box_features])

        return y
