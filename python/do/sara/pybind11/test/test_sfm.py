import unittest

import numpy as np

import imageio

import pysara_pybind11 as pysara


class TestPybind11(unittest.TestCase):

    def test_oeregion(self):
        f = pysara.OERegion()
        self.assertTrue(np.array_equiv(f.coords, np.zeros((1, 2),
                                                          dtype=np.float)))
        a = pysara.OERegion()
        b = pysara.OERegion()
        self.assertEqual(a, b)

    def test_compute_sift_keypoints(self):
        image = np.zeros((24, 32, 1), dtype=float)
        features, descriptors = pysara.compute_sift_keypoints(image)
