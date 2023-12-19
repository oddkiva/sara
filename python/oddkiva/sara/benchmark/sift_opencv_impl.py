import time
import numpy as np
import cv2


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s s' % (time.time() - self.tstart))


img = cv2.imread('/home/david/Desktop/Datasets/sfm/castle_int/0000.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()

for i in range(10):
    with Timer("OpenCV SIFT"):
        kp = sift.detect(gray, None)

print("keypoints = ", len(kp))
