import sys

import numpy as np

import matplotlib
# try:
#     matplotlib.use('Qt5Agg')
# except:
#     print('Failed to load Qt5 backend for matplotlib')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pysara_pybind11 import VideoStream


video_file = sys.argv[1]
video_stream = VideoStream()
video_stream.open(video_file)

video_frame = np.empty(video_stream.sizes(), dtype=np.uint8)


fig = plt.figure(facecolor='white')
im = plt.imshow(video_frame)


def update_fig(*args):
    video_stream.read(video_frame)
    im.set_data(video_frame)
    return im,


ani = animation.FuncAnimation(fig, update_fig, interval=5, blit=True)
plt.show()
