// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <stdint.h>

#include <gtest/gtest.h>

#include <DO/Core/Image/Operations.hpp>
#include <DO/Core/Pixel/Typedefs.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO;


TEST(TestImageConversion, test_find_min_max_for_1d_pixel)
{
  Image<int> image(10, 20);

  for (int y = 0; y < 20; ++y)
    for (int x = 0; x < 10; ++x)
      image(x, y) = x+y;

  int min, max;
  find_min_max(min, max, image);
  EXPECT_EQ(0, min);
  EXPECT_EQ(9+19, max);
}


TEST(TestImageConversion, test_find_min_max_for_3d_pixel)
{
  Image<Rgb8> image(10, 20);

  for (int y = 0; y < 20; ++y)
    for (int x = 0; x < 10; ++x)
      image(x, y).fill(x+y);

  Rgb8 min, max;
  find_min_max(min, max, image);
  EXPECT_EQ(Rgb8(0, 0, 0), min);
  EXPECT_EQ(Rgb8(28, 28, 28), max);
}


TEST(TestImageConversion, test_smart_image_conversion)
{
  Image<Rgb8> rgb8_image(2, 2);
  for (int y = 0; y < rgb8_image.height(); ++y)
    for (int x = 0; x < rgb8_image.width(); ++x)
      rgb8_image(x, y).fill(255);

  Image<float> float_image = rgb8_image.convert<float>();
  for (int y = 0; y < float_image.height(); ++y)
    for (int x = 0; x < float_image.width(); ++x)
      EXPECT_EQ(float_image(x, y), 1.f);

  Image<int> int_image = float_image.convert<int>();
  for (int y = 0; y < int_image.height(); ++y)
    for (int x = 0; x < int_image.width(); ++x)
      EXPECT_EQ(int_image(x, y), INT_MAX);

  Image<Rgb64f> rgb64f_image = int_image.convert<Rgb64f>();
  for (int y = 0; y < rgb64f_image.height(); ++y)
    for (int x = 0; x < rgb64f_image.width(); ++x)
      EXPECT_MATRIX_NEAR(rgb64f_image(x, y), Vector3d::Ones(), 1e-7);
}


TEST(TestImageConversion, test_image_color_rescale)
{
  Image<Rgb32f> rgb_image(10, 10);
  for (int y = 0; y < rgb_image.height(); ++y)
    for (int x = 0; x < rgb_image.width(); ++x)
      rgb_image(x, y).fill(static_cast<float>(x+y));

  rgb_image = color_rescale(rgb_image);
  Rgb32f rgb_min, rgb_max;
  find_min_max(rgb_min, rgb_max, rgb_image);
  EXPECT_EQ(rgb_min, Vector3f::Zero());
  EXPECT_EQ(rgb_max, Vector3f::Ones());

  Image<float> float_image(10, 10);
  for (int y = 0; y < float_image.height(); ++y)
    for (int x = 0; x < float_image.width(); ++x)
      float_image(x, y) = static_cast<float>(x+y);

  float_image = color_rescale(float_image);
  float float_min, float_max;
  find_min_max(float_min, float_max, float_image);
  EXPECT_EQ(float_min, 0);
  EXPECT_EQ(float_max, 1);
}


// ========================================================================== //
// Run the tests.
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
