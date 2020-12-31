import unittest

import numpy as np

import imageio

from do import sara


class TestPybind11(unittest.TestCase):

    def test_oeregion(self):
        f = sara.OERegion()
        self.assertTrue(np.array_equiv(f.coords, np.zeros((1, 2),
                                                          dtype=np.float)))
        a = sara.OERegion()
        b = sara.OERegion()
        self.assertEqual(a, b)

    def test_compute_sift_keypoints(self):
        image = np.zeros((24, 32), dtype=float)
        features, descriptors = sara.compute_sift_keypoints(image)
