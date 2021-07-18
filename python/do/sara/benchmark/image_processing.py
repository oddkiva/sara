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
    gradx = np.empty(video_stream.sizes()[:2], dtype=np.float32)
    grady = np.empty(video_stream.sizes()[:2], dtype=np.float32)
    mag = np.empty(video_stream.sizes()[:2], dtype=np.float32)
    ori = np.empty(video_stream.sizes()[:2], dtype=np.float32)

    ed = sara.EdgeDetector()

    sigma = 5.
    gauss_trunc = 3
    ksize = int(2 * sigma * gauss_trunc + 1)

    # Benchmarking on 4K video against Vanilla OpenCV.
    # Should compare with OpenCV CUDA implementation.

    sara.create_window(w, h)

    while video_stream.read(video_frame):
        # Shakti wins.
        with sara.Timer("[SHAKTI] rgb8 to gray32f"):
            shakti.convert_rgb8_to_gray32f_cpu(video_frame, video_frame_gray8)
        with sara.Timer("[OPENCV] rgb8 to gray32f"):
            cv2.cvtColor(video_frame, cv2.COLOR_RGB2GRAY, video_frame_gray8)
            video_frame_gray32f = video_frame_gray8.astype(np.float32) / 255
        print()

        # On CUDA-architecture: Shakti by almost a factor 2.
        # On Macbook Air: OpenCV, CPU-GPU transfer may be the bottleneck.
        with sara.Timer("[SHAKTI] gaussian convolution CPU"):
            shakti.gaussian_convolution(video_frame_gray32f,
                                        video_frame_convolved, sigma,
                                        gauss_trunc, False)
        with sara.Timer("[SHAKTI] gaussian convolution GPU"):
            shakti.gaussian_convolution(video_frame_gray32f,
                                        video_frame_convolved, sigma,
                                        gauss_trunc, True)
        with sara.Timer("[OPENCV] gaussian convolution"):
            cv2.GaussianBlur(video_frame_gray32f, (ksize, ksize), sigma,
                             video_frame_convolved)
        print()

        # OpenCV wins with a small edge but it also does more arithmetic
        # operations because of Sobel. So the comparison was not very fair.
        with sara.Timer("[SHAKTI] gradient"):
            shakti.gradient_2d_32f(video_frame_gray32f, gradx, grady)
        with sara.Timer("[OPENCV] (sobel) gradient"):
            cv2.Sobel(video_frame_gray32f, cv2.CV_32F, 1, 0, gradx)
            cv2.Sobel(video_frame_gray32f, cv2.CV_32F, 0, 1, grady)
        print()

        # Shakti wins on CUDA-architecture by almost a factor 2.
        with sara.Timer("[SHAKTI] polar gradient"):
            shakti.polar_gradient_2d_32f(video_frame_gray32f, mag, ori)
        with sara.Timer("[OPENCV] polar gradient"):
            cv2.Sobel(video_frame_gray32f, cv2.CV_32F, 1, 0, gradx)
            cv2.Sobel(video_frame_gray32f, cv2.CV_32F, 0, 1, grady)
            cv2.cartToPolar(gradx, grady, mag, ori)
        print()

        with sara.Timer("[SHAKTI] gray32f to rgb8"):
            shakti.convert_gray32f_to_rgb8_cpu(video_frame_convolved,
                                               video_frame)

        sara.draw_image(video_frame)

if __name__ == '__main__':
    sara.run_graphics(user_main)
