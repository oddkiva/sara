// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <gtest/gtest.h>

#include <DO/Sara/Geometry/Objects.hpp>

#include "TestPolygon.hpp"


using namespace std;
using namespace DO::Sara;


class TestTriangle : public TestPolygon {};

TEST_F(TestTriangle, test_constructor_and_area_computation)
{
  Triangle t1(Point2d(0,0), Point2d(100, 0), Point2d(100, 100));
  EXPECT_NEAR(area(t1), 1e4/2., 1e-10);

  Triangle t2(Point2d(100,0), Point2d(0, 0), Point2d(100, 100));
  EXPECT_NEAR(signed_area(t2), -1e4/2., 1e-10);
}

TEST_F(TestTriangle, test_point_inside_triangle)
{
  Triangle t(Point2d(-3, -3), Point2d(4, -2), Point2d(0, 4));

  double exact_area = area(t);
  int pixel_area = sweep_count_pixels([&](Point2d& p) {
    return inside(p, t);
  });

  double relError = fabs(exact_area - pixel_area) / exact_area;
  EXPECT_NEAR(relError, 0., 5e-2);
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}