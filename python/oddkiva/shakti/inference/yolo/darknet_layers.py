from typing import Any

import torch.nn as nn
import torch.nn.functional as F


class MaxPool(nn.Module):

    def __init__(self, kernel_size, stride):
        super(MaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
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


class ConvBNA(nn.Module):

    def __init__(self, in_channels, darknet_params: dict[str, Any], id: int):
        super(ConvBNA, self).__init__()
        self.block = nn.Sequential()

        # Unpack the block parameters from the Darknet configuration.
        batch_normalize = darknet_params['batch_normalize']
        out_channels = darknet_params['filters']
        kernel_size = darknet_params['size']
        stride = darknet_params['stride']
        add_padding = darknet_params['pad']
        activation = darknet_params['activation']
        pad_size = (kernel_size - 1) // 2 if add_padding else 0

        # Add the convolutional layer
        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding=pad_size,
            bias=not batch_normalize,
            padding_mode='zeros' # Let's be explicit about the padding value
        )
        self.block.add_module(f'conv{id}', conv)

        # Add the batch-normalization layer
        if batch_normalize:
            self.block.add_module(f'batch_norm{id}', nn.BatchNorm2d(out_channels))

        # Add the activation layer
        if activation == 'leaky':
            activation_fn = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation == 'mish':
            activation_fn = nn.Mish()
        elif activation == 'linear':
            activation_fn = nn.Identity()
        elif activation == 'logistic':
            activation_fn = nn.Sigmoid()
        else:
            raise ValueError(f'No convolutional activation named {activation}')
        self.block.add_module(f'{activation}{id}', activation_fn);


    def forward(self, x):
        return self.block.forward(x)


class Route(nn.Module):

    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x):
        raise NotImplementedError()
        # layers = block['layers'].split(',')
        # layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
        # if len(layers) == 1:
        #     if 'groups' not in block.keys() or int(block['groups']) == 1:
        #         x = outputs[layers[0]]
        #         outputs[ind] = x
        #     else:
        #         groups = int(block['groups'])
        #         group_id = int(block['group_id'])
        #         _, b, _, _ = outputs[layers[0]].shape
        #         x = outputs[layers[0]][:, b // groups * group_id:b // groups * (group_id + 1)]
        #         outputs[ind] = x
        # elif len(layers) == 2:
        #     x1 = outputs[layers[0]]
        #     x2 = outputs[layers[1]]
        #     x = torch.cat((x1, x2), 1)
        #     outputs[ind] = x
        # elif len(layers) == 4:
        #     x1 = outputs[layers[0]]
        #     x2 = outputs[layers[1]]
        #     x3 = outputs[layers[2]]
        #     x4 = outputs[layers[3]]
        #     x = torch.cat((x1, x2, x3, x4), 1)
        #     outputs[ind] = x
        # else:
        #     print("rounte number > 2 ,is {}".format(len(layers)))
        return x


class Shortcut(nn.Module):

    def __init__(self):
        super(Shortcut, self).__init__()

    def forward(self, x):
        raise NotImplementedError()
        # from_layer = int(block['from'])
        # activation = block['activation']
        # from_layer = from_layer if from_layer > 0 else from_layer + ind
        # x1 = outputs[from_layer]
        # x2 = outputs[ind - 1]
        # x = x1 + x2
        # if activation == 'leaky':
        #     x = F.leaky_relu(x, 0.1, inplace=True)
        # elif activation == 'relu':
        #     x = F.relu(x, inplace=True)
        # outputs[ind] = x
        return x


class Yolo(nn.Module):

    def __init__(self):
        super(Yolo, self).__init__()

    def forward(self, x):
        raise NotImplementedError()
        return x;
