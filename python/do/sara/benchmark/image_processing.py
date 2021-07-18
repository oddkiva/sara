import sys

from PySide2.QtCore import Qt

import numpy as np

import do.sara as sara
import do.shakti as shakti

import cv2


def user_main():
    video_file = sys.argv[1]
    video_stream = sara.VideoStream()
    video_stream.open(video_file)

    h, w, _ = video_stream.sizes()

    video_frame = np.empty(video_stream.sizes(), dtype=np.uint8)
    video_frame_gray8 = np.empty(video_stream.sizes()[:2], dtype=np.uint8)
    video_frame_gray32f = np.empty(video_stream.sizes()[:2], dtype=np.float32)
    video_frame_convolved = np.empty(video_stream.sizes()[:2], dtype=np.float32)

    ed = sara.EdgeDetector()

    sigma = 1.6
    gauss_trunc = 4

    while video_stream.read(video_frame):
        with sara.Timer("[SHAKTI] rgb8 to gray32f"):
            shakti.convert_rgb8_to_gray32f_cpu(video_frame, video_frame_gray8)

        with sara.Timer("[OPENCV] rgb8 to gray32f"):
            cv2.cvtColor(video_frame, cv2.COLOR_RGB2GRAY, video_frame_gray8)
            video_frame_gray32f = video_frame_gray8.astype(np.float32) / 255
        print()

        with sara.Timer("[SHAKTI] gaussian convolution"):
            shakti.gaussian_convolution(video_frame_gray32f,
                                        video_frame_convolved, sigma, 4)

        with sara.Timer("[OPENCV] gaussian convolution"):
            cv2.GaussianBlur(video_frame_gray32f, None, sigma,
                             video_frame_convolved)
        print()

if __name__ == '__main__':
    sara.run_graphics(user_main)
