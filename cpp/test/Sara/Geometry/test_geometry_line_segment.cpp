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

#define BOOST_TEST_MODULE "Geometry/Objects/Line Segments"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Geometry/Objects/LineSegment.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_line_segment_intersection)
{
  const auto s1 = LineSegment({-1, 0}, {1, 0});
  const auto s2 = LineSegment({0, -1}, {0, 1});

  auto p = Eigen::Vector2d{};
  BOOST_CHECK(intersection(s1, s2, p));
  BOOST_CHECK_SMALL(p.norm(), 1e-12);
}
