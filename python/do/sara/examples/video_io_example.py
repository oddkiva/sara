import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from do.sara import VideoStream


matplotlib.use('Qt5Agg')

print("hello")
video_stream = VideoStream()
video_stream.open(('/home/david/Desktop/GitHub/DO-CV/sara/examples/VideoIO/'
                  'orion_1.mpg'))

print("video_sizes = {}".format(video_stream.sizes()))
video_frame = np.empty(video_stream.sizes(), dtype=np.uint8)

fig = plt.figure(facecolor='white')
im = plt.imshow(video_frame)


def update_fig(*args):
    video_stream.read(video_frame)
    im.set_data(video_frame)
    return im


ani = animation.FuncAnimation(fig, update_fig, interval=5, blit=True)
plt.show()
