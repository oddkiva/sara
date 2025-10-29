# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import time
import numpy as np
from io import BytesIO
from PIL import Image

import oddkiva.sara as sara
import oddkiva.shakti as shakti


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print(f'[{self.name}]')
        print(f'Elapsed: {time.time() - self.tstart} s')



img = np.asarray(Image.open('/home/david/Desktop/Datasets/sfm/castle_int/0000.png'))
gray32f = np.empty(img.shape[:2], dtype=np.float32)
shakti.convert_rgb8_to_gray32f_cpu(img, gray32f)

scale_count_per_octave = 3

for i in range(10):
    with Timer("[SARA] Feature detection"):
        kp = sara.compute_sift_keypoints(
            image=gray32f,
            pyramid_params=sara.ImagePyramidParams(
                first_octave_index=0,
                scale_count_per_octave=scale_count_per_octave + 3,
                scale_geometric_factor=2. ** (1. / scale_count_per_octave),
                image_padding_size=1,
                scale_camera=1,
                scale_initial=1.6
            ),
            gauss_truncate=3.,
            extremum_thres=0.01,
            edge_ratio_thres=10.,
            extremum_refinement_iter=3,
            parallel=True)
