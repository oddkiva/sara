import pathlib
import unittest
from os import path

import numpy as np

import imageio

import pysara_pybind11 as pysara


class TestVideoStream(unittest.TestCase):

    def test_me(self):
        video_stream = pysara.VideoStream()

        video_stream.open(path.join(str(pathlib.Path.home()),
                                    'GitLab/DO-CV',
                                    'sara/cpp/examples/Sara/VideoIO',
                                    'orion_1.mpg'))

        video_frame = np.zeros(video_stream.sizes(), dtype=np.uint8)

        video_stream.read(video_frame)


if __name__ == '__main__':
    unittest.main()
