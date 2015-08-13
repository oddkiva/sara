// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
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

#include <DO/Sara/Core/Meta.hpp>
#include <DO/Sara/Core/Pixel/Typedefs.hpp>


using namespace std;
using namespace DO::Sara;


TEST(TestPixelTypedefs, test_3d_colors_typedefs)
{
  // Color3XX with unsigned integer types.
  ASSERT_TRUE(bool(std::is_same<Color3ub, Matrix<unsigned char, 3, 1> >::value));
  ASSERT_TRUE(bool(std::is_same<Color3us, Matrix<unsigned short, 3, 1> >::value));
  ASSERT_TRUE(bool(std::is_same<Color3ui, Matrix<unsigned int, 3, 1> >::value));

  // Color3X with signed integer types.
  ASSERT_TRUE(bool(std::is_same<Color3b, Matrix<char, 3, 1> >::value));
  ASSERT_TRUE(bool(std::is_same<Color3s, Matrix<short, 3, 1> >::value));
  ASSERT_TRUE(bool(std::is_same<Color3i, Matrix<int, 3, 1> >::value));

  // Color3X with floating-point types.
  ASSERT_TRUE(bool(std::is_same<Color3f, Matrix<float, 3, 1> >::value));
  ASSERT_TRUE(bool(std::is_same<Color3d, Matrix<double, 3, 1> >::value));
}


TEST(TestPixelTypedefs, test_4d_colors_typedefs)
{
  // Color4XX with unsigned integer types.
  ASSERT_TRUE(bool(std::is_same<Color4ub, Matrix<unsigned char, 4, 1> >::value));
  ASSERT_TRUE(bool(std::is_same<Color4us, Matrix<unsigned short, 4, 1> >::value));
  ASSERT_TRUE(bool(std::is_same<Color4ui, Matrix<unsigned int, 4, 1> >::value));

  // Color4X with signed integer types.
  ASSERT_TRUE(bool(std::is_same<Color4b, Matrix<char, 4, 1> >::value));
  ASSERT_TRUE(bool(std::is_same<Color4s, Matrix<short, 4, 1> >::value));
  ASSERT_TRUE(bool(std::is_same<Color4i, Matrix<int, 4, 1> >::value));

  // Color4X with floating-point types.
  ASSERT_TRUE(bool(std::is_same<Color4f, Matrix<float, 4, 1> >::value));
  ASSERT_TRUE(bool(std::is_same<Color4d, Matrix<double, 4, 1> >::value));
}


TEST(TestPixelTypedefs, test_rgb_color_constants)
{
  // Check colors with signed char channels.
  EXPECT_EQ(Color3b(127, -128, -128), red<char>());
  EXPECT_EQ(Color3b(-128, 127, -128), green<char>());
  EXPECT_EQ(Color3b(-128, -128, 127), blue<char>());
  EXPECT_EQ(Color3b(-128, 127, 127), cyan<char>());
  EXPECT_EQ(Color3b(127, -128, 127), magenta<char>());
  EXPECT_EQ(Color3b(127, 127, -128), yellow<char>());
  EXPECT_EQ(Color3b(-128, -128, -128), black<char>());

  // Check colors with unsigned char channels.
  EXPECT_EQ(Color3ub(255, 0, 0), Red8);
  EXPECT_EQ(Color3ub(0, 255, 0), Green8);
  EXPECT_EQ(Color3ub(0, 0, 255), Blue8);
  EXPECT_EQ(Color3ub(0, 255, 255), Cyan8);
  EXPECT_EQ(Color3ub(255, 0, 255), Magenta8);
  EXPECT_EQ(Color3ub(255, 255, 0), Yellow8);
  EXPECT_EQ(Color3ub(0, 0, 0), Black8);

  // Check colors
  EXPECT_EQ(Color3f(1, 0, 0), red<float>());
  EXPECT_EQ(Color3f(0, 1, 0), green<float>());
  EXPECT_EQ(Color3f(0, 0, 1), blue<float>());
  EXPECT_EQ(Color3f(0, 1, 1), cyan<float>());
  EXPECT_EQ(Color3f(1, 0, 1), magenta<float>());
  EXPECT_EQ(Color3f(1, 1, 0), yellow<float>());
  EXPECT_EQ(Color3f(0, 0, 0), black<float>());
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
