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

#define BOOST_TEST_MODULE "Geometry/Algorithms/Region Inner Boundary"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Geometry/Algorithms.hpp>

#include "AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestRegionInnerBoundary)

BOOST_AUTO_TEST_CASE(test_compute_region_inner_boundary)
{
  auto regions = Image<int>{5, 5};
  regions.matrix() <<
      0, 0, 1, 2, 3,
      0, 1, 2, 2, 3,
      0, 2, 2, 2, 2,
      4, 4, 2, 2, 2,
      4, 4, 2, 2, 5;

  const auto true_boundaries = vector<vector<Point2i>>{
      {Point2i{0, 2}, Point2i{0, 1}, Point2i{0, 0}, Point2i{1, 0}},
      {Point2i{2, 0}, Point2i{1, 1}},
      {Point2i{3, 0}, Point2i{2, 1}, Point2i{1, 2}, Point2i{2, 3},
       Point2i{2, 4}, Point2i{3, 4}, Point2i{4, 3}, Point2i{4, 2},
       Point2i{3, 1}},
      {Point2i{4, 0}, Point2i{4, 1}},
      {Point2i{0, 3}, Point2i{1, 3}, Point2i{0, 4}, Point2i{1, 4}},
      {Point2i{4, 4}}};

  const auto actual_boundaries = compute_region_inner_boundaries(regions);

  BOOST_CHECK_EQUAL(true_boundaries.size(), actual_boundaries.size());
  for (auto i = 0u; i < true_boundaries.size(); ++i)
    BOOST_REQUIRE_ITEMS_EQUAL(true_boundaries[i], actual_boundaries[i]);
}

BOOST_AUTO_TEST_SUITE_END()
