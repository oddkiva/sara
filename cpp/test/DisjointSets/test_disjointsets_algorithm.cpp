// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <gtest/gtest.h>

#include <unordered_set>

#include <DO/Sara/DisjointSets/AdjacencyList.hpp>
#include <DO/Sara/DisjointSets/DisjointSets.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestDisjointSets, test_on_image)
{
  auto regions = Image<int>{ 5, 5 };
  regions.matrix() <<
//  0  1  2  3  4
    0, 0, 1, 2, 3,
//  5  6  7  8  9
    0, 1, 1, 2, 3,
// 10 11 12 13 14
    0, 2, 2, 2, 2,
// 15 16 17 18 19
    4, 4, 2, 2, 2,
// 20 21 22 23 24
    4, 4, 2, 2, 5;

  // Compute the adjacency list using the 4-connectivity.
  auto adjacency_list = compute_adjacency_list_2d(regions);
  auto disjoint_sets = DisjointSets{ regions.size(), adjacency_list };
  disjoint_sets.compute_connected_components();
  auto components = disjoint_sets.get_connected_components();
  for (auto& component : components)
    sort(component.begin(), component.end());

  auto true_components = vector<vector<size_t>>{
    { 0, 1, 5, 10 },
    { 2, 6, 7 },
    { 3, 8, 11, 12, 13, 14, 17, 18, 19, 22, 23 },
    { 4, 9 },
    { 15, 16, 20, 21 },
    { 24 },
  };

  EXPECT_EQ(components.size(), true_components.size());
  for (size_t i = 0; i < components.size(); ++i)
    EXPECT_NE(components.end(),
              find(components.begin(), components.end(), true_components[i]));
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
