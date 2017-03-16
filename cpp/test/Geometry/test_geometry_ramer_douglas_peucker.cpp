// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE                                                      \
  "Geometry/Algorithms/Ramer Douglas Peucker Polygon Simplification Algorithm"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Geometry/Algorithms.hpp>

#include "../AssertHelpers.hpp"

#include "TestPolygon.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestRamerDouglasPeucker)

BOOST_AUTO_TEST_CASE(test_squared_distance)
{
  const auto a = Point2d{0., 0.};
  const auto b = Point2d{0., 10.};
  const auto x = Point2d{2., 8.};

  BOOST_CHECK_CLOSE(detail::orthogonal_distance(a, b, x), 2., 1e-6);
}

BOOST_AUTO_TEST_CASE(test_squared_distance_2)
{
  const auto a = Point2d{0., 0.};
  const auto b = Point2d{10., 10.};
  const auto x = Point2d{5., 3.};

  BOOST_CHECK_CLOSE(detail::orthogonal_distance(a, b, x), sqrt(2.), 1e-6);
}

BOOST_AUTO_TEST_CASE(test_linesegment_simplification)
{
  const auto line = vector<Point2d>{Point2d(191, 639), Point2d(192, 639)};

  BOOST_CHECK(line == ramer_douglas_peucker(line, 0.1));
}

BOOST_AUTO_TEST_CASE(test_polylines_simplification)
{
  const auto polylines = vector<Point2d>{Point2d(0, 0), Point2d(1, 0.25),
                                         Point2d(2, 0.5), Point2d(9, 0)};

  const auto expected_polylines =
      vector<Point2d>{Point2d(0, 0), Point2d(2, 0.5), Point2d(9, 0)};

  BOOST_CHECK(expected_polylines == ramer_douglas_peucker(polylines, 0.1));
}

BOOST_AUTO_TEST_CASE(test_square)
{
  const auto square = vector<Point2d>{
      Point2d(0, 0), Point2d(0.25, 0), Point2d(0.5, 0), Point2d(0.75, 0),
      Point2d(1, 0), Point2d(1, 1),    Point2d(0, 1),   Point2d(0, 0)};

  const auto expected_polygon = vector<Point2d>{Point2d(0, 0), Point2d(1, 0),
                                                Point2d(1, 1), Point2d(0, 1)};

  BOOST_CHECK(expected_polygon == ramer_douglas_peucker(square, 0.1));
}

BOOST_AUTO_TEST_SUITE_END()
