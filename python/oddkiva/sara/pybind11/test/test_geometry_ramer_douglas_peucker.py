# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from pysara_pybind11 import ramer_douglas_peucker
import numpy as np


def test_square():
    square = [(0, 0), (0.25, 0), (0.5, 0), (0.75, 0), (1, 0), (1, 1),
              (0, 1), (0, 0)]

    actual_polygon = ramer_douglas_peucker(square, 0.1)
    actual_polygon = [tuple(a.astype(int))
                      for a in actual_polygon]

    expected_polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]

    assert expected_polygon == actual_polygon
