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
  BBox bbox(p1, p2);
  EXPECT_EQ(bbox.topLeft(), Point2d(a, a));
  EXPECT_EQ(bbox.topRight(), Point2d(b, a));
  EXPECT_EQ(bbox.bottomRight(), Point2d(b, b));
  EXPECT_EQ(bbox.bottomLeft(), Point2d(a, b));

  auto predicate = [&](const Point2d& p) {
    return inside(p, bbox);
  };
  auto groundTruth = [&](const Point2d& p) {
    return 
      p.cwiseMin(p1) == p1 && 
      p.cwiseMax(p2) == p2;
  };
  sweep_check(predicate, groundTruth);
}

TEST_F(TestBBox, test_constructor_from_point_set)
{
  Point2d points[] = {
    Point2d::Zero(),
    Point2d(a, a),
    center
  };

  BBox bbox(points, points+3);
  EXPECT_EQ(bbox.topLeft(), points[0]);
  EXPECT_EQ(bbox.bottomRight(), points[2]);
}

TEST_F(TestBBox, test_point_inside_bbox)
{
  BBox bbox(p1, p2);

  Point2d points[] = {
    Point2d::Zero(),
    Point2d(a, a),
    center
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