// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015-2017 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Geometry/Objects/CSG"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Geometry/Objects/CSG.hpp>
#include <DO/Sara/Geometry/Objects/Cone.hpp>
#include <DO/Sara/Geometry/Objects/Ellipse.hpp>

#include "TestPolygon.hpp"


using namespace std;
using namespace DO::Sara;


class TestFixtureForCSG : public TestFixtureForPolygon
{
public:
  TestFixtureForCSG()
    : TestFixtureForPolygon()
  {
    _width = 15;
    _height = 15;
  }
};


BOOST_FIXTURE_TEST_SUITE(TestCSG, TestFixtureForCSG)

BOOST_AUTO_TEST_CASE(test_intersection_ellipse_cone)
{
  const auto E = Ellipse{12., 12., 0., Point2d::Zero()};
  const auto K = AffineCone2{Point2d{1, 0}, Point2d{0, 1}, Point2d::Zero()};

  const auto ell = CSG::Singleton<Ellipse>{E};
  const auto cone = CSG::Singleton<AffineCone2>{K};
  const auto inter = ell * cone;

  auto estimated_area = static_cast<double>(
      sweep_count_pixels([&](const Point2d& p) { return inter.contains(p); }));

  const auto true_area = area(E) / 4.;
  BOOST_CHECK_CLOSE(estimated_area, true_area, 16. /* percent */);
}

BOOST_AUTO_TEST_SUITE_END()
