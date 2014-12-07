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


class TestBBox : public TestPolygon {};


TEST_F(TestBBox, test_constructor_and_accessors)
{
  BBox bbox(_p1, _p2);
  EXPECT_EQ(bbox.top_left(), Point2d(_a, _a));
  EXPECT_EQ(bbox.top_right(), Point2d(_b, _a));
  EXPECT_EQ(bbox.bottom_right(), Point2d(_b, _b));
  EXPECT_EQ(bbox.bottom_left(), Point2d(_a, _b));

  auto predicate = [&](const Point2d& p) {
    return inside(p, bbox);
  };
  auto groundTruth = [&](const Point2d& p) {
    return 
      p.cwiseMin(_p1) == _p1 && 
      p.cwiseMax(_p2) == _p2;
  };
  sweep_check(predicate, groundTruth);
}

TEST_F(TestBBox, test_constructor_from_point_set)
{
  Point2d points[] = {
    Point2d::Zero(),
    Point2d(_a, _a),
    _center
  };

  BBox bbox(points, points+3);
  EXPECT_EQ(bbox.top_left(), points[0]);
  EXPECT_EQ(bbox.bottom_right(), points[2]);
}

TEST_F(TestBBox, test_point_inside_bbox)
{
  BBox bbox(_p1, _p2);

  Point2d points[] = {
    Point2d::Zero(),
    Point2d(_a, _a),
    _center
  };
  EXPECT_FALSE(inside(points[0], bbox));
  EXPECT_TRUE (inside(points[1], bbox));
  EXPECT_TRUE (inside(points[2], bbox));
}


int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}