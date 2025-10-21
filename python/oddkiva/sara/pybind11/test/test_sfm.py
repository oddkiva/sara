import numpy as np

import pysara_pybind11 as sara


def test_oeregion():
    f = sara.OERegion()
    assert np.array_equiv(f.coords, np.zeros((1, 2), dtype=np.float32))

    a = sara.OERegion()
    b = sara.OERegion()
    assert a == b

def test_compute_sift_keypoints():
    image = np.zeros((24, 32), dtype=float)
    keypoints = sara.compute_sift_keypoints(image,
                                            sara.ImagePyramidParams(),
                                            True)
    f, d = sara.features(keypoints), sara.descriptors(keypoints)
