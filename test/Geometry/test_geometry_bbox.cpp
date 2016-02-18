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

#include <gtest/gtest.h>

#include <DO/Sara/Geometry/Objects.hpp>

#include "TestPolygon.hpp"

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


class TestBBox : public TestPolygon
{
};

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

  vector<Point2d> points_vector(points, points+3);
  BBox bbox2(points_vector);
  EXPECT_MATRIX_EQ(bbox.top_left(), bbox2.top_left());
  EXPECT_MATRIX_EQ(bbox.bottom_right(), bbox2.bottom_right());
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

TEST_F(TestBBox, test_bbox_ostream)
{
  const auto bbox = BBox{};

  stringstream buffer;
  CoutRedirect cout_redirect{ buffer.rdbuf() };
  cout << bbox << endl;

  auto text = buffer.str();

  EXPECT_NE(text.find("top-left: ["), string::npos);
  EXPECT_NE(text.find("bottom-right: ["), string::npos);
}

TEST_F(TestBBox, test_bbox_degenerate)
{
  const auto bbox = BBox{};
  EXPECT_TRUE(degenerate(bbox));
}

TEST_F(TestBBox, test_intersection)
{
  const auto b1 = BBox{ Point2d{ 0, 0 }, Point2d{ 1, 1 } };
  const auto b2 = BBox{ Point2d{ 0.5, 0.5 }, Point2d{ 1.5, 1.5 } };

  const auto inter = BBox{ Point2d{ 0.5, 0.5 }, Point2d{ 1, 1 } };
  EXPECT_EQ(inter, intersection(b1, b2));

  auto expected_jaccard_distance = 1 - area(inter) / (area(b1) + area(b2) - area(inter));
  EXPECT_EQ(expected_jaccard_distance, jaccard_distance(b1, b2));
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
