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

#define BOOST_TEST_MODULE "Geometry/Tools/Projective"

#include <DO/Sara/Geometry/Tools/Projective.hpp>

#include <boost/test/unit_test.hpp>

#include "AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_line)
{
  const Eigen::Vector2d a = 2 * Eigen::Vector2d::UnitX();
  const Eigen::Vector2d b = Eigen::Vector2d::UnitY();

  const auto line_expected = Eigen::Vector3d{-0.5, -1, 1};
  const auto line = Projective::line(a.homogeneous().eval(),  //
                                     b.homogeneous().eval());

  BOOST_CHECK_SMALL(line.cross(line_expected).norm(), 1e-9);
}
