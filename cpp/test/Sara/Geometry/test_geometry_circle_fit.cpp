// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Geometry/Algorithms/Circle Fit"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Geometry/Algorithms/CircleFit.hpp>

#include "AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_fit_circle_2d)
{
  // Find the unit circle.
  {
    auto pts = Eigen::Matrix<double, 3, 2> {};
    pts <<
                    1.0,               0.0,
                    0.0,               1.0,
      std::cos(M_PI/3.), std::sin(M_PI/3.);

    const auto circle = fit_circle_2d (pts);

    BOOST_CHECK_SMALL (circle.center.norm (), 1e-5);
    BOOST_CHECK_CLOSE (circle.radius, 1., 1e-5);
  }

  // Find the circle with center (10, 10) and with radius 5.
  {
    auto pts = Eigen::Matrix<double, 3, 2> {};
    pts <<
                    1.0,               0.0,
                    0.0,               1.0,
      std::cos(M_PI/3.), std::sin(M_PI/3.);
    pts.array () = 5. * pts.array () + 10.;

    const auto circle = fit_circle_2d (pts);

    BOOST_CHECK_SMALL ((circle.center - Eigen::Vector2d (10, 10)).norm (), 1e-5);
    BOOST_CHECK_CLOSE (circle.radius, 5., 1e-5);
  }
}
