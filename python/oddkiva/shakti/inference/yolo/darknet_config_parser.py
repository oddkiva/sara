from pathlib import Path


class DarknetConfigParser:

    def __init__(self):
        self._lines = None
        self._metadata = None
        self._model = None

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
        pad = bool(conv_params['pad'])
        activation = bool(conv_params['activation'])
        # The following parameter has default values.
        batch_normalize = int(conv_params.get('batch_normalize', '0'))

        self._model[layer_index] = {
            'convolutional': {
                'batch_normalize': bool(batch_normalize),
                'filters': filters,
                'size': size,
                'pad': pad,
            }
        }

    def typify_route_parameters(self, layer_index):
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

        groups = int(route_params['groups'])
        group_id = int(route_params['group_id'])

        self._model[layer_index] = {
            'route': {
                'layers': layers,
            }
        }

    def read(self, path: Path):
        self.read_lines(path)
        self.parse_lines()

        for layer_index in range(len(self._model)):
            layer_name = list(self._model[layer_index].keys())[0]
            if layer_name == 'convolutional':
                self.typify_convolutional_parameters(layer_index)
