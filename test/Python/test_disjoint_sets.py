import sys

import numpy as np

from unittest import TestCase

import do.sara as sara


# class TestDisjointSets(TestCase):
# 
#     def test_disjoint_sets(self):
A = np.array(
    [[0, 1],
     [0, 1]],
    dtype=np.int32)

print("hello")
import ipdb; ipdb.set_trace()
print(sara.compute_adjacency_list_2d(A))
