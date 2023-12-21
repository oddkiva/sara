import logging
from pathlib import Path

import numpy as np


logging.basicConfig(level=logging.DEBUG)


class NetworkWeightLoader:

    def __init__(self, model_weight_path: Path):
        with open(model_weight_path, 'rb') as fp:
            # The following also works and we keep it commented out as a
            # reminder.
            # self.major = struct.unpack('i', fp.read(4))[0]
            # self.minor = struct.unpack('i', fp.read(4))[0]
            # self.revision = struct.unpack('i', fp.read(4))[0]

            self.major = np.fromfile(fp, count=1, dtype=np.int32)[0]
            self.minor = np.fromfile(fp, count=1, dtype=np.int32)[0]
            self.revision = np.fromfile(fp, count=1, dtype=np.int32)[0]

            if self.major * 10 + self.minor >= 2:
                logging.debug("byte size of seen = 64 bits")
                self.seen = np.fromfile(fp, count=1, dtype=np.uint64)[0]
            else:
                logging.debug("byte size of seen= 32 bits")
                self.seen = np.fromfile(fp, count=1, dtype=np.uint32)
            self.transpose = (self.major > 1000) or (self.minor > 1000)

            logging.debug(f'version = {self.major}.{self.minor}.{self.revision}')
            logging.debug(f'transpose = {self.transpose}')
            logging.debug(f'model has seen {self.seen} images')
            logging.debug(f'trained with {int(self.seen / 1000)} K-images '
                          f'({int(self.seen / 64000)} K-batch of 64 images)')

            self._weights = np.fromfile(fp, dtype=np.float32)
            self._cursor = 0

    def read(self, num_elements: int) -> np.ndarray:
        logging.debug(f'cursor = {self._cursor}')
        i1 = self._cursor
        i2 = i1 + num_elements
        logging.debug(f'[i1, i2[ = [{i1}, {i2}[')
        x = self._weights[i1:i2]
        self._cursor += num_elements
        return x
