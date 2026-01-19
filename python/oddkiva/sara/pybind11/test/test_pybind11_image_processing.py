# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import pysara_pybind11 as sara

import numpy as np


def test_resize():
    src = np.arange(16).reshape((4, 4))
    src = np.array([src, src, src]).astype(np.float32)

    dst = np.zeros((3, 4, 4)).astype(np.float32)
    
    sara.resize(src, dst)

    assert np.linalg.norm(src - dst) < 1e-12
