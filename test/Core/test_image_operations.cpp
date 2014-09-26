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


// ========================================================================== //
// Run the tests.
int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}