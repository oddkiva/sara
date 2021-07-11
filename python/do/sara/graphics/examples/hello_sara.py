import sys

from PySide2.QtCore import Qt

import numpy as np

from skimage.io import imread

import do.sara as sara

from pysara_pybind11 import VideoStream


def user_main():
    video_file = sys.argv[1]
    video_stream = VideoStream()
    video_stream.open(video_file)

    video_frame = np.empty(video_stream.sizes(), dtype=np.uint8)
    h, w, _ = video_stream.sizes()

    sara.create_window(w, h)

    for y in range(20):
        for x in range(20):
            sara.draw_point(x, y, (255, 0, 0))
            sara.millisleep(1)

    key = sara.get_key()
    if key == Qt.Key_Escape:
        return

    while video_stream.read(video_frame):
        sara.draw_image(video_frame)

if __name__ == '__main__':
    sara.run_graphics(user_main)
