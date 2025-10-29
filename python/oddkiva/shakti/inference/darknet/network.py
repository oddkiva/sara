# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import logging
from pathlib import Path
from typing import Optional

import numpy as np

import torch
import torch.nn as nn

import oddkiva.shakti.inference.darknet as darknet
import oddkiva.shakti.inference.darknet.v4 as v4


logging.basicConfig(level=logging.DEBUG)


class Network(nn.Module):

    def __init__(self, cfg: darknet.Config, up_to_layer: Optional[int]=None):
        super(Network, self).__init__()

        input_shape = (
            None,
            cfg._metadata['channels'],
            cfg._metadata['height'],
            cfg._metadata['width']
        )
        self.in_shape_at_block = [input_shape]
        self.out_shape_at_block = [input_shape]
        self.up_to_layer = up_to_layer
        self.model = self.create_network(cfg)

    def input_shape(self):
        return self.in_shape_at_block[0]

    def create_network(self, cfg: darknet.Config):
        model = nn.ModuleList()

        if cfg._model is None:
            raise ValueError()

        logging.debug(cfg._metadata)

        conv_id = 0
        route_id = 0
        shortcut_id = 0
        max_pool_id = 0
        upsample_id = 0
        yolo_id = 0

        for (i, block) in enumerate(cfg._model):
            if self.up_to_layer is not None and i > self.up_to_layer:
                break

            layer_name = list(block.keys())[0]
            layer_params = block[layer_name]

            if layer_name == 'convolutional':
                self._append_conv(model, layer_params, conv_id)
                conv_id += 1
            elif layer_name == 'route':
                self._append_route(model, layer_params, route_id)
                route_id += 1
            elif layer_name == 'shortcut':
                self._append_shortcut(model, layer_params, shortcut_id)
                shortcut_id += 1
            elif layer_name == 'maxpool':
                self._append_max_pool(model, layer_params, max_pool_id)
                max_pool_id += 1
            elif layer_name == 'upsample':
                self._append_upsample(model, layer_params, upsample_id)
                upsample_id += 1
            elif layer_name == 'yolo':
                self._append_yolo(model, layer_params, yolo_id)
                yolo_id += 1
            else:
                raise NotImplementedError(
                    f'Pytorch layer "{layer_name}" is not implemented'
                )

        return model

    def load_convolutional_weights(self,
                                   weights_file: Path,
                                   version: str='v4'):
        if version != 'v4':
            raise NotImplementedError

        weight_loader = v4.NetworkWeightLoader(weights_file)

        for block_idx, block in enumerate(self.model):
            if self.up_to_layer is not None and block_idx > self.up_to_layer:
                break
            if type(block) is not darknet.ConvBNA:
                continue

            logging.debug(f'[LOADING] {block_idx} {block}')

            conv = block.layers[0]

            # Read in the following order.
            # 1. Convolution bias weights
            self._load_weights(conv.bias, weight_loader)

            # 2. BN weights
            if block.batch_normalize:
                if block.fuse_conv_bn_layer:
                    shape = (conv.weight.shape[0],)
                    block.bn_weights['scale'] = self._read_weights(
                        shape, weight_loader)
                    block.bn_weights['running_mean'] = self._read_weights(
                        shape, weight_loader)
                    block.bn_weights['running_var'] = self._read_weights(
                        shape, weight_loader)
                else:
                    bn = block.layers[1]
                    self._load_weights(bn.weight, weight_loader)
                    self._load_weights(bn.running_mean, weight_loader)
                    self._load_weights(bn.running_var, weight_loader)

            # 3. Convolution weights.
            self._load_weights(conv.weight, weight_loader)

            # 4. Recalculate the weights and bias if we fuse.
            if block.fuse_conv_bn_layer:
                eps = .00001
                bn_mean = block.bn_weights['running_mean']
                bn_var = block.bn_weights['running_var']
                bn_scale = block.bn_weights['scale']

                factor = bn_mean / np.sqrt(bn_var + eps)
                conv.bias.data -= torch.from_numpy(bn_scale * factor)
                for c_out in range(conv.weight.shape[0]):
                    scale_w = bn_scale[c_out] / np.sqrt(bn_var[c_out] + eps)
                    conv.weight.data[c_out, :, :, :] *= scale_w

        logging.debug(f'weight loader cursor = {weight_loader._cursor}')
        logging.debug(f'weights num elements = {weight_loader._weights.size}')
        if self.up_to_layer is None:
            assert weight_loader._cursor == weight_loader._weights.size

    def _read_weights(self, shape, weight_loader):
        return weight_loader.read(shape[0]).reshape(shape)

    def _load_weights(self, block_params, weight_loader):
        params_as_np = weight_loader\
            .read(block_params.numel())\
            .reshape(block_params.data.shape)
        block_params.data = torch.from_numpy(params_as_np)

    def save_weights(self, weights_file: Path):
        raise NotImplementedError

    def _append_conv(self, model, layer_params, conv_id):
        # Extract the input shape.
        shape_in = self.out_shape_at_block[-1]

        # Calculate the output shape.
        n, c_in, h_in, w_in = shape_in
        stride = layer_params['stride']
        c_out = layer_params['filters']
        h_out, w_out = h_in // stride, w_in // stride
        shape_out = (n, c_out, h_out, w_out)

        # Store.
        self.in_shape_at_block.append(shape_in)
        self.out_shape_at_block.append(shape_out)

        # Append the convolutional block to the model.
        model.append(darknet.ConvBNA(c_in, layer_params, conv_id))
        logging.debug(
            f'[Conv {conv_id}] '
            f'{shape_in} -> {shape_out}'
        )
        logging.debug(f'{layer_params}')

    def _append_route(self, model, layer_params, route_id):
        layers = layer_params['layers']
        groups = layer_params['groups']
        group_id = layer_params['group_id']

        if len(layers) == 1:
            # Extract the input shape
            shape_in = self.out_shape_at_block[layers[0]]

            # Calculate the output shape
            n, c_in, h_in, w_in = shape_in
            shape_out = (n, c_in // groups, h_in, w_in)

            # Store.
            self.in_shape_at_block.append(shape_in)
            self.out_shape_at_block.append(shape_out)

            # Append the route-slice block.
            model.append(darknet.RouteSlice(
                layers[0], groups, group_id, route_id))
            logging.debug(
                f'[Route {route_id}] (Slide): '
                f'{shape_in} -> {shape_out}'
            )
        else:
            # Fetch all the input shapes.
            shape_ins = [self.out_shape_at_block[l] for l in layers]

            # Calculate the output shape.
            n, _, h_in, w_in = shape_ins[0]
            c_out = sum([shape_in[1] for shape_in in shape_ins])
            shape_out = (n, c_out, h_in, w_in)

            # Store.
            self.in_shape_at_block.append(shape_ins)
            self.out_shape_at_block.append(shape_out)

            # Append the route-concat block.
            if len(layers) == 2:
                model.append(darknet.RouteConcat2(layers, route_id))
            elif len(layers) == 4:
                model.append(darknet.RouteConcat4(layers, route_id))
            else:
                raise NotImplementedError(
                    "Route-Concat supports only 2 or 4 inputs")
            logging.debug(
                f'[Route {route_id}] (Concat): '
                f'{shape_ins} -> {shape_out}\n'
                f'                    layers = {layers}'
            )

    def _append_shortcut(self, model, layer_params, shortcut_id):
        # Fetch all the input shapes.
        from_layer = layer_params['from']
        activation = layer_params['activation']
        layers = [-1, from_layer] 
        shape_ins = [self.out_shape_at_block[l] for l in layers]

        # Calculate the output shape.
        assert shape_ins[0][2:] == shape_ins[1][2:]
        n, c_in, h_in, w_in = shape_ins[0]
        shape_out = (n, c_in, h_in, w_in)

        # Store.
        self.in_shape_at_block.append(shape_ins)
        self.out_shape_at_block.append(shape_out)

        model.append(darknet.Shortcut(from_layer, activation))
        logging.debug(
            f'[Shortcut {shortcut_id}] {shape_ins} -> {shape_out}\n'
            f'                         from_layer = {from_layer}'
        )

    def _append_max_pool(self, model, layer_params, max_pool_id):
        # Extract the input shape
        shape_in = self.out_shape_at_block[-1]

        # Calculate the output shape.
        size = layer_params['size']
        stride = layer_params['stride']
        n, c_in, h_in, w_in = shape_in
        shape_out = (n, c_in, h_in // stride, w_in // stride)

        # Store.
        self.in_shape_at_block.append(shape_in)
        self.out_shape_at_block.append(shape_out)

        model.append(darknet.MaxPool(size, stride))
        logging.debug(f'[MaxPool {max_pool_id}] '
                      f'{shape_in} -> {shape_out}')

    def _append_upsample(self, model, layer_params, upsample_id):
        # Extract the input shape
        shape_in = self.out_shape_at_block[-1]

        # Calculate the output shape.
        stride = layer_params['stride']
        n, c_in, h_in, w_in = shape_in
        shape_out = (n, c_in, h_in * stride, w_in * stride)

        # Store.
        self.in_shape_at_block.append(shape_in)
        self.out_shape_at_block.append(shape_out)

        model.append(darknet.Upsample(stride))
        logging.debug(f'[Upsample {upsample_id}] '
                      f'{shape_in} -> {shape_out}')

    def _append_yolo(self, model, layer_params, yolo_id):
        # Extract the input shape
        shape_in = self.out_shape_at_block[-1]

        # Calculate the output shape.
        shape_out = shape_in

        # Store.
        self.in_shape_at_block.append(shape_in)
        self.out_shape_at_block.append(shape_out)

        model.append(darknet.Yolo(layer_params))
        logging.debug(f'[YOLO {yolo_id}] '
                      f'{shape_in} -> {shape_out}')

    def _forward(self, x):
        ys = [x]
        boxes = []
        for block in self.model:
            if type(block) is darknet.ConvBNA:
                conv = block
                x = ys[-1]
                y = conv.forward(x)
                ys.append(y)
            elif type(block) is darknet.MaxPool:
                maxpool = block
                x = ys[-1]
                y = maxpool(x)
                ys.append(y)
            elif type(block) is darknet.RouteSlice:
                slice = block
                y_idx = slice.layer if slice.layer < 0 else slice.layer + 1
                x = ys[y_idx]
                y = slice(x)
                ys.append(y)
            elif type(block) is darknet.RouteConcat2:
                concat = block

                ids = [l if l < 0 else l + 1 for l in concat.layers]
                xs = [ys[i] for i in ids]
                y = concat(*xs)
                ys.append(y)
            elif type(block) is darknet.RouteConcat4:
                concat = block

                ids = [l if l < 0 else l + 1 for l in concat.layers]
                xs = [ys[i] for i in ids]
                y = concat(*xs)
                ys.append(y)
            elif type(block) is darknet.Shortcut:
                shortcut = block
                i1 = -1
                if shortcut.from_layer < 0:
                    i2 = shortcut.from_layer
                else:
                    i2 = shortcut.from_layer + 1
                xs = [ys[i] for i in [i1, i2]]
                y = shortcut(*xs)
                ys.append(y)

            elif type(block) is darknet.Upsample:
                upsample = block
                x = ys[-1]
                y = upsample(x)
                ys.append(y)
            elif type(block) is darknet.Yolo:
                yolo = block
                x = ys[-1]
                y = yolo(x)
                ys.append(y)
                boxes.append(y)
            else:
                raise NotImplementedError

        return ys, boxes

    def forward(self, x):
        ys, boxes = self._forward(x)
        if self.up_to_layer is None:
            return tuple(boxes)
        else:
            return ys[-1]
