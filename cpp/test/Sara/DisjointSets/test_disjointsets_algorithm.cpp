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

#define BOOST_TEST_MODULE "DisjointSets/Algorithms"

#include <unordered_set>

#include <boost/test/unit_test.hpp>

#include <DO/Sara/DisjointSets/AdjacencyList.hpp>
#include <DO/Sara/DisjointSets/DisjointSets.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestDisjointSets)

BOOST_AUTO_TEST_CASE(test_on_image)
{
  auto regions = Image<int>{5, 5};
  // clang-format off
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
  // clang-format on

  auto adj_list_data = compute_adjacency_list_2d(regions);
  const auto adj_list = AdjacencyList{adj_list_data};

  // Compute the adjacency list using the 4-connectivity.
  auto disjoint_sets = DisjointSets{};
  disjoint_sets.compute_connected_components(adj_list);

  auto components = disjoint_sets.get_connected_components();
  for (auto& component : components)
    sort(component.begin(), component.end());

  auto true_components = vector<vector<size_t>>{
    {0, 1, 5, 10},
    {2, 6, 7},
    {3, 8, 11, 12, 13, 14, 17, 18, 19, 22, 23},
    {4, 9},
    {15, 16, 20, 21},
    {24},
  };

  BOOST_CHECK_EQUAL(components.size(), true_components.size());
  for (size_t i = 0; i < components.size(); ++i)
    BOOST_REQUIRE(
      find(components.begin(), components.end(), true_components[i]) !=
      components.end());
}

BOOST_AUTO_TEST_SUITE_END()
