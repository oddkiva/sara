from unittest import TestCase

import numpy as np

from do.sara import compute_adjacency_list_2d, compute_connected_components


class TestDisjointSets(TestCase):

    def test_compute_adjacency_list_2d(self):
        regions = np.array([[0, 1], [0, 1]], dtype=np.int32)

        adj_list = compute_adjacency_list_2d(regions)
        self.assertEqual(adj_list, [[2], [3], [0], [1]])

    def test_disjoint_sets(self):
        regions = np.array([[0, 0, 1, 2, 3],
                            [0, 1, 1, 2, 3],
                            [0, 2, 2, 2, 2],
                            [4, 4, 2, 2, 2],
                            [4, 4, 2, 2, 5]],
                           dtype=np.int32)
        components = compute_connected_components(regions)
        self.assertEqual(
            [[0, 1, 5, 10],
             [2, 6, 7],
             [3, 8, 11, 12, 13, 14, 17, 18, 19, 22, 23],
             [4, 9],
             [15, 16, 20, 21],
             [24]],
            components)
