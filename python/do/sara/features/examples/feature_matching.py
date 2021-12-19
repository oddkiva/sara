import sys

from PySide2.QtCore import Qt

import numpy as np

import do.sara as sara
import do.shakti as shakti


def draw_feature(image, kp, color, pen_width=2):
    r = kp.radius() * np.sqrt(2.)
    o = kp.orientation
    a = kp.coords
    b = a + r * np.array([np.cos(o), np.sin(o)])
    sara.image_draw.draw_circle(image, a, r, (0, 0, 0), pen_width + 2)
    sara.image_draw.draw_line(image, a, b, (0, 0, 0), pen_width + 2)
    sara.image_draw.draw_circle(image, a, r, color, pen_width)
    sara.image_draw.draw_line(image, a, b, color, pen_width)

def user_main():
    video_file = sys.argv[1]
    video_stream = sara.VideoStream()
    video_stream.open(video_file)

    h, w, _ = video_stream.sizes()

    video_frame = np.empty(video_stream.sizes(), dtype=np.uint8)
    video_frame_gray32f = np.empty(video_stream.sizes()[:2], dtype=np.float32)

    sara.create_window(w, h)

    # Feature detection and matching parameters.
    first_octave = 0
    image_pyramid_params = sara.ImagePyramidParams(
        first_octave_index=first_octave)
    sift_nn_ratio = 0.8

    # Work data.
    kp_prev = None
    kp_curr = None

    f = 0
    while video_stream.read(video_frame):
        with sara.Timer("[SHAKTI] RGB8 to Gray32f"):
            shakti.convert_rgb8_to_gray32f_cpu(video_frame, video_frame_gray32f)

        with sara.Timer("[SARA] Feature detection"):
            kp_prev = kp_curr
            kp_curr = sara.compute_sift_keypoints(video_frame_gray32f,
                                                  image_pyramid_params,
                                                  True)

        if kp_prev is None:
            continue;
        with sara.Timer("[SARA] Feature matching"):
            ann_matcher = sara.AnnMatcher(kp_prev, kp_curr, 0.8)
            matches = ann_matcher.compute_matches()

        with sara.Timer("[SARA] Draw"):
            f1 = [f for f in sara.features(kp_prev)]
            f2 = [f for f in sara.features(kp_curr)]
            for f in f1[:1000]:
                draw_feature(video_frame, f, (127, 0, 0))
            for f in f2[:1000]:
                draw_feature(video_frame, f, (255, 0, 0))
            for m in matches[:100]:
                x = f1[m.x].coords
                y = f2[m.y].coords
                sara.image_draw.draw_line(video_frame, x, y, (255, 255, 0), 2)
            sara.draw_image(video_frame)


if __name__ == '__main__':
    sara.run_graphics(user_main)
