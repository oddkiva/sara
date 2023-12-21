from pathlib import Path

import torch.nn as nn

import oddkiva.shakti.inference.darknet as darknet


class Network(nn.Module):

    def __init__(self, cfg: darknet.Config):
        super(Network, self).__init__()

        input_shape = (
            None,
             cfg._metadata['channels'],
             cfg._metadata['height'],
             cfg._metadata['width']
        )
        self.in_shape_at_block = [input_shape]
        self.out_shape_at_block = [input_shape]
        self.model = self.create_network(cfg)

    def create_network(self, cfg: darknet.Config):
        model = nn.ModuleList()

        if cfg._model is None:
            raise ValueError()

        print(cfg._metadata)

        conv_id = 0
        route_id = 0
        max_pool_id = 0
        upsample_id = 0
        yolo_id = 0

        for block in cfg._model:
            layer_name = list(block.keys())[0]
            layer_params = block[layer_name]

            if layer_name == 'convolutional':
                self._append_conv(model, layer_params, conv_id)
                conv_id += 1
            elif layer_name == 'route':
                self._append_route(model, layer_params, route_id)
                route_id += 1
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
        print(f'[Conv {conv_id}] '
              f'{self.in_shape_at_block[-1]} -> {self.out_shape_at_block[-1]}')

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
            print(f'[Route {route_id}] (Slide): '
                  f'{self.in_shape_at_block[-1]} -> {self.out_shape_at_block[-1]}')
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
            model.append(darknet.RouteConcat(layers, route_id))
            print(f'[Route {route_id}] (Concat): '
                  f'{self.in_shape_at_block[-1]} -> {self.out_shape_at_block[-1]}')
            print(f'                    layers = {layers}')

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
        print(f'[MaxPool {max_pool_id}] '
              f'{self.in_shape_at_block[-1]} -> {self.out_shape_at_block[-1]}')

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
        print(f'[Upsample {upsample_id}] '
              f'{self.in_shape_at_block[-1]} -> {self.out_shape_at_block[-1]}')

    def _append_yolo(self, model, layer_params, yolo_id):
        # Extract the input shape
        shape_in = self.out_shape_at_block[-1]

        # Calculate the output shape.
        shape_out = shape_in

        # Store.
        self.in_shape_at_block.append(shape_in)
        self.out_shape_at_block.append(shape_out)

        model.append(darknet.Yolo(layer_params))
        print(f'[YOLO {yolo_id}] '
              f'{self.in_shape_at_block[-1]} -> {self.out_shape_at_block[-1]}')

    def load_convolutional_weights(self, conv, weights_file: Path):
        pass
        # with open(weights_file, 'rb') as fp:
        #     fp.read(

    def save_weights(self, weights_file: Path):
        pass
