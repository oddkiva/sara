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

#include <DO/ImageProcessing/Interpolation.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO;


TEST(TestInterpolation, test_interpolation)
{
  Image<float> f(2, 2);
  f.matrix() <<
    0, 1,
    0, 1;
  double value;

  for (int x = 0; x < 2; ++x)
  {
    for (int y = 0; y < 2; ++y)
    {
      Vector2d p = Vector2i(x, y).cast<double>();
      value = interpolate(f, p);
      ASSERT_NEAR(f(x, y), value, 1e-7);
    }
  }

  value = interpolate(f, Vector2d(0.5, 0.0));
  ASSERT_NEAR(0.5, value, 1e-7);

  value = interpolate(f, Vector2d(0.5, 0.2));
  ASSERT_NEAR(0.5, value, 1e-7);

  value = interpolate(f, Vector2d(0.5, 0.1));
  ASSERT_NEAR(0.5, value, 1e-7);

  value = interpolate(f, Vector2d(0.5, 0.8));
  ASSERT_NEAR(0.5, value, 1e-7);

  value = interpolate(f, Vector2d(0.5, 1.));
  ASSERT_NEAR(0.5, value, 1e-7);

  f.matrix() <<
    0, 0,
    1, 1;
  value = interpolate(f, Vector2d(0.0, 0.5));
  ASSERT_NEAR(0.5, value, 1e-7);

  value = interpolate(f, Vector2d(0.2, 0.5));
  ASSERT_NEAR(0.5, value, 1e-7);

  value = interpolate(f, Vector2d(0.5, 0.5));
  ASSERT_NEAR(0.5, value, 1e-7);

  value = interpolate(f, Vector2d(0.8, 0.5));
  ASSERT_NEAR(0.5, value, 1e-7);

  value = interpolate(f, Vector2d(1, 0.5));
  ASSERT_NEAR(0.5, value, 1e-7);
}


TEST(TestInterpolation, test_interpolation_2)
{
  Image<float> f(2, 2);
  f.matrix() <<
    0, 1,
    1, 2;
  double value;

  value = interpolate(f, Vector2d(1, 1));
  ASSERT_NEAR(2, value, 1e-7);
}


int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}