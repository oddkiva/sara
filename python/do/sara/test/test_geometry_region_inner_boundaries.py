from unittest import TestCase

import numpy as np

from do.sara import compute_region_inner_boundaries


class TestRegionInnerBoundary(TestCase):

    def test_compute_region_inner_boundaries(self):
        regions = np.array([[0, 0, 1, 2, 3],
                            [0, 1, 2, 2, 3],
                            [0, 2, 2, 2, 2],
                            [4, 4, 2, 2, 2],
                            [4, 4, 2, 2, 5]],
                           dtype=np.int32)

        true_boundaries = [
            {(0, 2), (0, 1), (0, 0), (1, 0)},
            {(2, 0), (1, 1)},
            {(3, 0), (2, 1), (1, 2), (2, 3), (2, 4), (3, 4), (4, 3), (4, 2),
             (3, 1)},
            {(4, 0), (4, 1)},
            {(0, 3), (1, 3), (0, 4), (1, 4)},
            {(4, 4)}
        ]

        actual_boundaries = compute_region_inner_boundaries(regions)

        # A boundary is an ordered set of vertices.
        actual_boundaries = [set(vertices) for vertices in actual_boundaries]

        self.assertItemsEqual(true_boundaries, actual_boundaries)
