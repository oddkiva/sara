// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <gtest/gtest.h>

#include <DO/Geometry/Objects.hpp>

#include "TestPolygon.hpp"


using namespace std;
using namespace DO;


class TestTriangle : public TestPolygon {};

TEST_F(TestTriangle, test_constructor_and_area_computation)
{
  Triangle t1(Point2d(0,0), Point2d(100, 0), Point2d(100, 100));
  EXPECT_NEAR(area(t1), 1e4/2., 1e-10);

  Triangle t2(Point2d(100,0), Point2d(0, 0), Point2d(100, 100));
  EXPECT_NEAR(signedArea(t2), -1e4/2., 1e-10);
}

TEST_F(TestTriangle, test_point_inside_triangle)
{
  Triangle t3(Point2d(50, 73), Point2d(350, 400), Point2d(25, 200));

  double exactArea3 = area(t3);
  int pixelArea3 = sweep_pixel_count([&](Point2d& p) {
    return inside(p, t3);
  });

  double relError = fabs(exactArea3 - pixelArea3)/exactArea3;
  EXPECT_NEAR(relError, 0., 5e-2);
}


int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}