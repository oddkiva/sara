import sys

import numpy as np

from unittest import TestCase

from do.sara import compute_adjacency_list_2d


class TestDisjointSets(TestCase):

    def test_compute_adjacency_list_2d(self):
        regions = np.array([[0, 1], [0, 1]], dtype=np.int32)

        adj_list = compute_adjacency_list_2d(regions)
        self.assertEqual(adj_list, [[2], [3], [0], [1]])
