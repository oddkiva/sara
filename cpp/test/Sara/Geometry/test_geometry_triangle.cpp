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

#define BOOST_TEST_MODULE "Geometry/Objects/Triangle"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Geometry/Objects.hpp>

#include "TestPolygon.hpp"


using namespace std;
using namespace DO::Sara;


class TestFixtureForTriangle : public TestFixtureForPolygon
{
};

BOOST_FIXTURE_TEST_SUITE(TestTriangle, TestFixtureForTriangle)

BOOST_AUTO_TEST_CASE(test_constructor_and_area_computation)
{
  Triangle t1(Point2d(0,0), Point2d(100, 0), Point2d(100, 100));
  BOOST_CHECK_CLOSE(area(t1), 1e4/2., 1e-3);

  Triangle t2(Point2d(100,0), Point2d(0, 0), Point2d(100, 100));
  BOOST_CHECK_CLOSE(signed_area(t2), -1e4/2., 1e-3);
}

BOOST_AUTO_TEST_CASE(test_point_inside_triangle)
{
  Triangle t(Point2d(0, 1), Point2d(4, 0), Point2d(0, 4));

  double exact_area = area(t);
  int pixel_area = sweep_count_pixels([&](Point2d& p) {
    return t.contains(p);
  });

  BOOST_CHECK_CLOSE(exact_area, pixel_area, 5e-2);
}

BOOST_AUTO_TEST_SUITE_END()
