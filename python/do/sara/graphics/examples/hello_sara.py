import sys

from PySide2.QtCore import Qt

import numpy as np

import do.sara as sara
import do.shakti as shakti


def user_main():
    video_file = sys.argv[1]
    video_stream = sara.VideoStream()
    video_stream.open(video_file)

    video_frame = np.empty(video_stream.sizes(), dtype=np.uint8)
    video_frame_gray = np.empty(video_stream.sizes()[:2], dtype=np.float32)
    video_frame_gray_convolved = np.empty(video_stream.sizes()[:2],
                                          dtype=np.float32)
    video_frame_display = np.empty(video_stream.sizes(), dtype=np.uint8)
    h, w, _ = video_stream.sizes()

    sara.create_window(w, h)
    sara.set_antialiasing(True)

    for y in range(20):
        for x in range(20):
            sara.draw_point(x, y, (255, 0, 0))
            sara.millisleep(1)

    sara.draw_line((100, 100), (200, 200), (0, 0, 127), 2)
    sara.draw_rect((100, 100), (100, 100), (0, 0, 127), 5)
    sara.draw_circle((100, 100), 50, (0, 255, 255), 1)
    sara.draw_ellipse((100, 100), 20, 90, 50, (0, 0, 255), 3)
    sara.draw_text((300, 300), "Hello Sara!",
                   (255, 0, 127), 20, -30, False, True, True)

    key = sara.get_key()
    if key == Qt.Key_Escape:
        return

    while video_stream.read(video_frame):
        with sara.Timer("convert rgb8 to gray32f"):
            shakti.convert_rgb8_to_gray32f_cpu(video_frame, video_frame_gray)
        with sara.Timer("Gaussian convolution"):
            shakti.gaussian_convolution(video_frame_gray,
                                        video_frame_gray_convolved, 1.6, 4)
        with sara.Timer("Convert gray32f to rgb8"):
            shakti.convert_gray32f_to_rgb8_cpu(video_frame_gray_convolved,
                                               video_frame_display)
        sara.draw_image(video_frame_display)

if __name__ == '__main__':
    sara.run_graphics(user_main)
