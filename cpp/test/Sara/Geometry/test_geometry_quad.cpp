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

#define BOOST_TEST_MODULE "Geometry/Objects/Quad"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Geometry/Objects.hpp>

#include "TestPolygon.hpp"


using namespace std;
using namespace DO::Sara;


class TestFixtureForQuad : public TestFixtureForPolygon
{
};

BOOST_FIXTURE_TEST_SUITE(TestQuad, TestFixtureForQuad)

BOOST_AUTO_TEST_CASE(test_constructor)
{
  const Point2d a{0, 0};
  const Point2d b{1, 0};
  const Point2d c{1, 1};
  const Point2d d{0, 1};

  const auto q1 = Quad{a, b, c, d};
  const auto q2 = Quad{BBox{a, c}};

  BOOST_CHECK(q1 == q2);
}

BOOST_AUTO_TEST_CASE(test_point_inside_quad)
{
  const auto bbox = BBox{_p1, _p2};
  const auto quad = Quad{bbox};

  BOOST_CHECK_CLOSE(area(bbox), area(quad), 1e-10);

  auto predicate = [&](const Point2d& p) { return quad.contains(p); };

  auto ground_truth = [&](const Point2d& p) {
    return _p1.x() <= p.x() && p.x() < _p2.x() && _p1.y() <= p.y() &&
           p.y() < _p2.y();
  };

  sweep_check(predicate, ground_truth);
}

BOOST_AUTO_TEST_SUITE_END()
