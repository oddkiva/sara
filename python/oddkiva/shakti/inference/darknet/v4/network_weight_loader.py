from pathlib import Path

import numpy as np


class NetworkWeightLoader:

    def __init__(self, model_weight_path: Path):
        with open(model_weight_path, 'rb') as fp:
            # self.major = struct.unpack('i', fp.read(4))[0]
            # self.minor = struct.unpack('i', fp.read(4))[0]
            # self.revision = struct.unpack('i', fp.read(4))[0]

            self.major = np.fromfile(fp, count=1, dtype=np.int32)[0]
            self.minor = np.fromfile(fp, count=1, dtype=np.int32)[0]
            self.revision = np.fromfile(fp, count=1, dtype=np.int32)[0]

            if self.major * 10 + self.minor >= 2:
                print("byte size of seen = 64 bits")
                self.seen = np.fromfile(fp, count=1, dtype=np.uint64)[0]
            else:
                print("byte size of seen= 32 bits")
                self.seen = np.fromfile(fp, count=1, dtype=np.uint32)
            self.transpose = (self.major > 1000) or (self.minor > 1000)

            print(f'version = {self.major}.{self.minor}.{self.revision}')
            print(f'transpose = {self.transpose}')
            print(f'model has seen {self.seen} images')
            print('trained with {} K-images ({} K-batch of 64 images)'.format(
                int(self.seen / 1000),
                int(self.seen / 64000)))

            self._weights = np.fromfile(fp, dtype=np.float32)
            self._cursor = 0

    def read(self, num_elements: int) -> np.ndarray:
        i1 = self._cursor
        i2 = i1 + num_elements
        x = self._weights[i1, i2]
        self._cursor += num_elements
        return x
