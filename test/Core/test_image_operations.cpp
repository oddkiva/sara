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


TEST(TestImageConversion, test_color_min_max_value)
{
  EXPECT_EQ(numeric_limits<int>::min(), color_min_value<int>());
  EXPECT_EQ(0, color_min_value<float>());

  EXPECT_EQ(std::numeric_limits<int>::max(), color_max_value<int>());
  EXPECT_EQ(1, color_max_value<float>());

  Matrix<uint8_t, 3, 1> expected_black = Matrix<uint8_t, 3, 1>::Zero();
  Matrix<uint8_t, 3, 1> actual_black = color_min_value<uint8_t, 3>();
  EXPECT_EQ(expected_black, actual_black);

  Vector3f expected_zeros = Vector3f::Zero();
  Vector3f actual_zeros = color_min_value<float, 3>();
  EXPECT_EQ(expected_zeros, actual_zeros);

  Matrix<uint8_t, 3, 1> expected_black_3 = Matrix<uint8_t, 3, 1>::Zero();
  Matrix<uint8_t, 3, 1> actual_black_3 = color_min_value<uint8_t, 3>();
  EXPECT_EQ(expected_black_3, actual_black_3);

  Vector3f expected_ones_3 = Vector3f::Zero();
  Vector3f actual_ones_3 = color_min_value<float, 3>();
  EXPECT_EQ(expected_ones_3, actual_ones_3);
}


TEST(TestImageConversion, test_image_conversion)
{
  Image<Rgb8> rgb8_image(10, 10);
  for (int y = 0; y < rgb8_image.height(); ++y)
    for (int x = 0; x < rgb8_image.width(); ++x)
      rgb8_image(x, y).fill(255);

  Image<Rgb32f> rgb32f_image;
  rgb32f_image = rgb8_image.convert_channel<Rgb32f>();
  for (int y = 0; y < rgb32f_image.height(); ++y)
    for (int x = 0; x < rgb32f_image.width(); ++x)
      EXPECT_EQ(rgb32f_image(x, y), Vector3f::Ones());

  Image<float> float_image;
  float_image = rgb8_image
    .convert_channel<Rgb32f>()
    .convert_color<float>();
  for (int y = 0; y < float_image.height(); ++y)
    for (int x = 0; x < float_image.width(); ++x)
      EXPECT_EQ(rgb32f_image(x, y), Vector3f::Ones());
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