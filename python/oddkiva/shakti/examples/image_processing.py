# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import sys

import numpy as np

import oddkiva.sara as sara
import oddkiva.shakti as shakti


def user_main():
    video_file = sys.argv[1]

    video_stream = sara.VideoStream()
    video_stream.open(video_file)
    sizes = video_stream.sizes()
    h, w, _ = sizes

    video_frame = np.empty(sizes, dtype=np.uint8)
    video_frame_gray32f = np.empty(sizes[:2], dtype=np.float32)
    video_frame_convolved = np.empty(sizes[:2], dtype=np.float32)
    mag = np.empty(sizes[:2], dtype=np.float32)
    ori = np.empty(sizes[:2], dtype=np.float32)

    sigma = 1.6
    gauss_trunc = 4

    sara.create_window(w, h)

    i = 0
    while True:
        with sara.Timer("Read"):
            if not video_stream.read(video_frame):
                break

        if i % 2 == 1:
            continue

        with sara.Timer("Processing"):
            shakti.convert_rgb8_to_gray32f_cpu(video_frame, video_frame_gray32f)
            shakti.gaussian_convolution(video_frame_gray32f,
                                        video_frame_convolved,
                                        sigma, gauss_trunc, False);
            shakti.polar_gradient_2d_32f(video_frame_convolved, mag, ori)

            mag = (mag + 1) * 0.5
            shakti.convert_gray32f_to_rgb8_cpu(mag, video_frame)

        sara.draw_image(video_frame)

if __name__ == '__main__':
    sara.run_graphics(user_main)
