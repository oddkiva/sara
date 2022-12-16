// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "DisjointSets/Algorithms"

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <unordered_map>

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/DisjointSets/DisjointSetsV2.hpp>


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

  const auto index = [](int x, int y) { return y * 5 + x; };

  // Compute the adjacency list using the 4-connectivity.
  auto disjoint_sets = v2::DisjointSets(5 * 5);
  for (auto y = 0; y < 5; ++y)
    for (auto x = 0; x < 5; ++x)
      BOOST_CHECK_EQUAL(index(x, y), disjoint_sets.parent(index(x, y)));

  omp_set_num_threads(omp_get_max_threads());

#pragma omp parallel for
  for (auto y = 0; y < 5; ++y)
  {
    for (auto x = 0; x < 5; ++x)
    {
      const auto me = index(x, y);
      for (auto dy = -1; dy <= 1; ++dy)
        for (auto dx = -1; dx <= 1; ++dx)
        {
          const auto x1 = x + dx;
          const auto y1 = y + dy;
          if (!(0 <= x1 && x1 < 5 && 0 <= y1 && y1 < 5))
            continue;

          if (dx == 0 && dy == 0)
            continue;

          if (regions(x, y) == regions(x1, y1))
            disjoint_sets.join(me, index(x1, y1));
        }
    }
  }

  auto components = std::map<int, std::vector<size_t>>{};
  for (auto y = 0; y < 5; ++y)
  {
    for (auto x = 0; x < 5; ++x)
    {
      const auto me = index(x, y);
      components[disjoint_sets.find_set(me)].push_back(me);
    }
  }
  for (auto& component : components)
    sort(component.second.begin(), component.second.end());

  for (const auto& [id, set] : components)
  {
    SARA_CHECK(id);
    for (const auto& i : set)
      std::cout << i << " ";
    std::cout << std::endl;
  }

  auto component_list = vector<vector<size_t>>{};
  for (auto& c : components)
    component_list.emplace_back(std::move(c.second));

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
    BOOST_REQUIRE(find(component_list.begin(), component_list.end(),
                       true_components[i]) != component_list.end());
}

BOOST_AUTO_TEST_SUITE_END()
