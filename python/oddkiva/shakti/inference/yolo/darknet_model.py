from pathlib import Path

import torch.nn as nn

from oddkiva.shakti.inference.yolo.darknet_config import DarknetConfig


class Darknet(nn.Module):

    def __init__(self, darknet_config: DarknetConfig):
        super(Darknet, self).__init__()
        self.create(self, darknet_config)

        self.model = self.create_network(darknet_config)

        self.major = None
        self.minor = None
        self.revision = None
        self.seen = None
        self.transpose = None

    def create_network(self, darknet_config: DarknetConfig):
        model = nn.ModuleList()

        if darknet_config._model is None:
            raise ValueError()

        for block in darknet_config._model:
            raise NotImplementedError(
                f'Block {block['type']} is not implemented'
            )

        return model

    def load_convolutional_weights(self, conv, weights_file: Path):
        with open(weights_file, 'rb') as fp:
            fp.read(

    def save_weights(self, weights_file: Path)
        pass
