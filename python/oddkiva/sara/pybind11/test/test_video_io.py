import pathlib
import numpy as np
import pysara_pybind11 as pysara


def test_me():
    video_stream = pysara.VideoStream()

    p = (pathlib.Path.home()/
         'GitLab/oddkiva' /
         'sara/cpp/examples/Sara/VideoIO' /
         'orion_1.mpg')
    video_stream.open(str(p), True)
    video_frame = np.zeros(video_stream.sizes(), dtype=np.uint8)
    video_stream.read(video_frame)
