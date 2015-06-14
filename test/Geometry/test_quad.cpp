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


class TestQuad : public TestPolygon {};


TEST_F(TestQuad, test_point_inside_quad)
{
  BBox bbox(_p1, _p2);
  Quad quad(bbox);

  EXPECT_NEAR(area(bbox), area(quad), 1e-10);

  auto predicate = [&](const Point2d& p) {
    return inside(p, quad);
  };
  auto groundTruth = [&](const Point2d& p) {
    return
      _p1.x() <= p.x() && p.x() < _p2.x() &&
      _p1.y() <= p.y() && p.y() < _p2.y() ;
  };
  sweep_check(predicate, groundTruth);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}