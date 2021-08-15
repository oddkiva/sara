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
    video_frame_gray32f = np.empty(video_stream.sizes()[:2], dtype=np.float32)

    sara.create_window(w, h)
    sara.set_antialiasing()

    kp_prev = None
    kp_curr = None

    while video_stream.read(video_frame):
        with sara.Timer("[SHAKTI] rgb8 to gray32f"):
            shakti.convert_rgb8_to_gray32f_cpu(video_frame, video_frame_gray32f)

        kp_prev = kp_curr
        kp_curr = sara.compute_sift_keypoints(video_frame_gray32f)

        if kp_prev is None:
            continue;
        ann_matcher = sara.AnnMatcher(kp_prev, kp_curr, 1.2)
        # matches = ann_matcher.compute_matches()
        import ipdb; ipdb.set_trace()


        sara.draw_image(video_frame)
        for kp in kp_prev[0]:
            sara.draw_circle(kp.coords, kp.radius() * np.sqrt(np.pi),
                             (127, 0, 0), 2)
        for kp in kp_curr[0]:
            sara.draw_circle(kp.coords, kp.radius() * np.sqrt(np.pi),
                             (255, 0, 0), 2)
        sara.millisleep(1)


if __name__ == '__main__':
    sara.run_graphics(user_main)
