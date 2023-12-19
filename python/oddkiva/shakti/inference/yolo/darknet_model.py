import torch.nn as nn

from oddkiva.shakti.inference.yolo.darknet_config import DarknetConfig


class Darknet(nn.Module):

    def __init__(self, darknet_config: DarknetConfig):
        super(Darknet, self).__init__()
        self.create(self, darknet_config)

        self.model = self.create_network(darknet_config)

    def create_network(self, darknet_config: DarknetConfig):
        model = nn.ModuleList()

        if darknet_config._model is None:
            raise ValueError()

        for block in darknet_config._model:
            raise NotImplementedError(
                f'Block {block['type']} is not implemented'
            )

        return model
