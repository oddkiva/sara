// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
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


struct CoutRedirect
{
  CoutRedirect(std::streambuf * new_buffer)
    : old(std::cout.rdbuf(new_buffer))
  {
  }

  ~CoutRedirect()
  {
    std::cout.rdbuf(old);
  }

private:
  std::streambuf * old;
};


class TestQuad : public TestPolygon {};


TEST_F(TestQuad, test_constructor)
{
  const Point2d a{ 0, 0 };
  const Point2d b{ 1, 0 };
  const Point2d c{ 1, 1 };
  const Point2d d{ 0, 1 };

  const auto q1 = Quad{ a, b, c, d };
  const auto q2 = Quad{ BBox{ a, c } };

  EXPECT_EQ(q1, q2);
}

TEST_F(TestQuad, test_point_inside_quad)
{
  const auto bbox = BBox{ _p1, _p2 };
  const auto quad = Quad{ bbox };

  EXPECT_NEAR(area(bbox), area(quad), 1e-10);

  auto predicate = [&](const Point2d& p) {
    return quad.contains(p);
  };

  auto ground_truth = [&](const Point2d& p) {
    return
      _p1.x() <= p.x() && p.x() < _p2.x() &&
      _p1.y() <= p.y() && p.y() < _p2.y() ;
  };

  sweep_check(predicate, ground_truth);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}