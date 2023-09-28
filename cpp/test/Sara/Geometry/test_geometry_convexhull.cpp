// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2017 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Geometry/Algorithms/Convex Hull"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Geometry/Algorithms/ConvexHull.hpp>

#include "AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestConvexHull)

BOOST_AUTO_TEST_CASE(test_quad_1)
{
  const auto points =
      vector<Point2d>{Point2d{0., 0.}, Point2d{1., 0.},   Point2d{1., 1.},
                      Point2d{0., 1.}, Point2d{0.5, 0.5}, Point2d{0.25, 0.25}};

  const auto expected_points = vector<Point2d>{
      Point2d{0., 0.}, Point2d{1., 0.}, Point2d{1., 1.}, Point2d{0., 1.}};

  const auto convex_hull = graham_scan_convex_hull(points);
  for (const auto& p : convex_hull)
    SARA_CHECK(p.transpose());
  BOOST_CHECK_EQUAL_COLLECTIONS(convex_hull.begin(), convex_hull.end(),
                                expected_points.begin(), expected_points.end());
}

BOOST_AUTO_TEST_CASE(test_quad_2)
{
  auto points = vector<Point2d>{};
  for (auto x = 0; x < 10; ++x)
    for (auto y = 0; y < 10; ++y)
      points.emplace_back(double(x), double(y));


  const auto expected_points = vector<Point2d>{
      Point2d{0., 0.}, Point2d{9., 0.}, Point2d{9., 9.}, Point2d{0., 9.}};

  const auto convex_hull = graham_scan_convex_hull(points);
  for (const auto& p : convex_hull)
    SARA_CHECK(p.transpose());
  BOOST_CHECK_EQUAL_COLLECTIONS(convex_hull.begin(), convex_hull.end(),
                                expected_points.begin(), expected_points.end());
}

BOOST_AUTO_TEST_SUITE_END()
