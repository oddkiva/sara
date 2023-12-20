from pathlib import Path

import torch.nn as nn

import oddkiva.shakti.inference.darknet as darknet


class Network(nn.Module):

    def __init__(self, cfg: darknet.Config):
        super(Network, self).__init__()

        self.model = self.create_network(cfg)

    def create_network(self, cfg: darknet.Config):
        model = nn.ModuleList()

        if cfg._model is None:
            raise ValueError()

        print(cfg._metadata)

        in_channels_at_block = [cfg._metadata['channels']]
        conv_id = 0
        route_id = 0
        max_pool_id = 0

        for block in cfg._model:
            layer_name = list(block.keys())[0]
            layer_params = block[layer_name]

            print(f'{in_channels_at_block}')

            if layer_name == 'convolutional':
                out_channels = layer_params['filters']
                stride = layer_params['stride']
                model.append(darknet.ConvBNA(in_channels_at_block[-1], layer_params,
                                             conv_id))
                print(f'[Conv{conv_id}]: Cin {in_channels_at_block[-1]}, Cout {out_channels}, stride {stride}')

                # Update the parameters for the next network block.
                in_channels_at_block.append(out_channels)
                conv_id += 1
            elif layer_name == 'route':
                layers = layer_params['layers']
                groups = layer_params['groups']
                group_id = layer_params['group_id']
                if len(layers) == 1:
                    model.append(darknet.RouteSlice(
                        layers[0], groups, group_id, route_id))

                else:
                    model.append(darknet.RouteConcat(layers, route_id))

                # Update t
                if len(layers) == 1:
                    out_channels = in_channels_at_block[-1] // groups
                    print(f'[Route{route_id}]: (Slide) Cin = {in_channels_at_block[-1]}, Cout {out_channels}')
                else:
                    out_channels = sum([in_channels_at_block[l-1] for l in layers])
                    print(
                        f'[Route{route_id}]: '
                        f'(Concat) Cin={in_channels_at_block[-1]}, Cout={out_channels}, layers={layers}'
                    )

                in_channels_at_block.append(out_channels)
                route_id += 1
            elif layer_name == 'maxpool':
                raise NotImplementedError
            else:
                raise NotImplementedError(
                    f'Pytorch layer "{layer_name}" is not implemented'
                )

        return model

    def load_convolutional_weights(self, conv, weights_file: Path):
        pass
        # with open(weights_file, 'rb') as fp:
        #     fp.read(

    def save_weights(self, weights_file: Path):
        pass
