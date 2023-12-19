from pathlib import Path
from typing import Any, Optional, TypeAlias


KeyValueStore: TypeAlias = dict[str, Any]


class DarknetConfig:

    def __init__(self):
        self._lines: Optional[list[str]] = None
        self._metadata: Optional[KeyValueStore] = None
        self._model: Optional[list[KeyValueStore]] = None

    def _is_comment(self, line: str):
        return line[0] == '#'

    def _is_section(self, line: str):
        return line[0] == '[' and line[-1] == ']'

    def _section_name(self, line: str):
        return line[1:-1]

    def read_lines(self, path: Path):
        with open(path, 'r') as fp:
            self._lines = fp.readlines()
            # Trim lines
            self._lines = [
                line.strip(' \n') for line in self._lines
            ]
            # Remove blank lines and comments.
            self._lines = [
                line for line in self._lines
                if line and not self._is_comment(line)
            ]

    def parse_lines(self):
        if self._lines is None:
            raise ValueError('lines is None')

        sections = []

        section_name = None
        for line in self._lines:
            if self._is_comment(line):
                continue
            elif self._is_section(line):
                section_name = self._section_name(line)
                section_props = {}
                sections.append({section_name: section_props})
            else:
                key, value = [l.strip(' ') for l in line.split('=')]
                sections[-1][section_name][key] = value

        self._metadata = sections[0]
        self._model = sections[1:]

    def typify_convolutional_parameters(self, layer_index):
        if self._model is None:
            raise ValueError('Model is None!')

        section = self._model[layer_index]

        section_name = list(section.keys())[0]
        if section_name != 'convolutional':
            raise RuntimeError('Not a convolutional layer!')

        conv_params = section[section_name]
        print(conv_params)

        # The following parameters must be present in the config file.
        filters = int(conv_params['filters'])
        size = int(conv_params['size'])
        stride = int(conv_params['stride'])
        pad = int(conv_params['pad'])
        activation = conv_params['activation']
        # The following parameter has default values.
        batch_normalize = int(conv_params.get('batch_normalize', '0'))

        self._model[layer_index] = {
            'convolutional': {
                'batch_normalize': bool(batch_normalize),
                'filters': filters,
                'size': size,
                'stride': stride,
                'pad': pad,
                'activation': activation,
            }
        }

    def typify_route_parameters(self, layer_index):
        if self._model is None:
            raise ValueError('Model is None!')

        section = self._model[layer_index]

        section_name = list(section.keys())[0]
        if section_name != 'route':
            raise RuntimeError('Not a route layer!')

        route_params = section[section_name]
        print(route_params)

        # The following parameters must be present in the config file.
        layers_str = route_params['layers']
        layers = layers_str.split(',')
        layers = [int(v.strip()) for v in layers]

        groups = int(route_params.get('groups', 1))
        group_id = int(route_params.get('group_id', -1))

        self._model[layer_index] = {
            'route': {
                'layers': layers,
                'groups': groups,
                'group_id': group_id
            }
        }

    def typify_maxpool_parameters(self, layer_index):
        if self._model is None:
            raise ValueError('Model is None!')

        section = self._model[layer_index]

        section_name = list(section.keys())[0]
        if section_name != 'maxpool':
            raise RuntimeError('Not a maxpool layer!')

        maxpool_params = section[section_name]
        print(maxpool_params)

        # The following parameters must be present in the config file.
        size = int(maxpool_params['size'])
        stride = int(maxpool_params['stride'])

        self._model[layer_index] = {
            'maxpool': {
                'size': size,
                'stride': stride,
            }
        }

    def typify_upsample_parameters(self, layer_index):
        if self._model is None:
            raise ValueError('Model is None!')

        section = self._model[layer_index]

        section_name = list(section.keys())[0]
        if section_name != 'upsample':
            raise RuntimeError('Not an upsample layer!')

        upsample_params = section[section_name]
        print(upsample_params)

        # The following parameters must be present in the config file.
        stride = int(upsample_params['stride'])

        self._model[layer_index] = {
            'upsample': {
                'stride': stride,
            }
        }

    def typify_yolo_parameters(self, layer_index):
        if self._model is None:
            raise ValueError('Model is None!')

        section = self._model[layer_index]

        section_name = list(section.keys())[0]
        if section_name != 'yolo':
            raise RuntimeError('Not a YOLO layer!')

        yolo_params = section[section_name]
        print(yolo_params)

        mask = [int(v.strip()) for v in yolo_params['mask'].split(',')]

        anchors = [int(v.strip()) for v in yolo_params['anchors'].split(',')]
        anchors_x = anchors[0::2]
        anchors_y = anchors[1::2]
        anchors = [(x, y) for (x, y) in zip(anchors_x, anchors_y)]

        classes = int(yolo_params['classes'])

        num = int(yolo_params['num'])
        jitter = float(yolo_params['jitter'])
        scale_x_y = float(yolo_params['scale_x_y'])
        cls_normalizer = float(yolo_params['cls_normalizer'])
        iou_normalizer = float(yolo_params['iou_normalizer'])
        iou_loss = yolo_params['iou_loss']
        ignore_thresh = yolo_params['ignore_thresh']
        truth_thresh = yolo_params['truth_thresh']
        random = yolo_params['random']
        resize = float(yolo_params['resize'])
        nms_kind = yolo_params['nms_kind']
        beta_nms = float(yolo_params['beta_nms'])

        # The following parameters must be present in the config file.
        self._model[layer_index] = {
            'upsample': {
                'mask': mask,
                'anchors': anchors,
                'classes': classes,
                'num': num,
                'jitter': jitter,
                'scale_x_y': scale_x_y,
                'cls_normalizer': cls_normalizer,
                'iou_normalizer': iou_normalizer,
                'iou_loss': iou_loss,
                'ignore_thresh': ignore_thresh,
                'truth_thresh': truth_thresh,
                'random': random,
                'resize': resize,
                'nms_kind': nms_kind,
                'beta_nms': beta_nms,
            }
        }

    def read(self, path: Path):
        self.read_lines(path)
        self.parse_lines()

        if self._model is None:
            raise ValueError('Model is None!')

        for layer_index in range(len(self._model)):
            layer_name = list(self._model[layer_index].keys())[0]
            if layer_name == 'convolutional':
                self.typify_convolutional_parameters(layer_index)
            elif layer_name == 'route':
                self.typify_route_parameters(layer_index)
            elif layer_name == 'maxpool':
                self.typify_maxpool_parameters(layer_index)
            elif layer_name == 'upsample':
                self.typify_upsample_parameters(layer_index)
            elif layer_name == 'yolo':
                self.typify_yolo_parameters(layer_index)
