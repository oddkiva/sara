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

#include <limits>
#include <stdint.h>
#include <vector>

#include <gtest/gtest.h>

#include "AssertHelpers.hpp"
#include "pixel.hpp"
#include "color_conversion.hpp"


using namespace std;
using namespace DO;


// ========================================================================== //
// Define the set of integral channel types, which we will test.
typedef testing::Types<float, double> FloatingPointChannelTypes;


// ========================================================================== //
// Define the parameterized test case.
template <typename ChannelType>
class TestConvertColorConversion : public testing::Test {};
TYPED_TEST_CASE_P(TestConvertColorConversion);


// ========================================================================== //
// RGB <-> RGBA.
TEST(TestConvertColorConversion, test_rgb_to_rgba)
{
    Pixel<double, Rgb> rgb(1,1,1);
    Pixel<double, Rgba> rgba;
    convert_color(rgb, rgba);
    EXPECT_EQ(Vector4d(1,1,1,1), rgba);
}

TEST(TestConvertColorConversion, test_rgba_to_rgb)
{
    Pixel<double, Rgba> rgba(1,1,1,1);
    Pixel<double, Rgb> rgb;
    convert_color(rgba, rgb);
    EXPECT_EQ(Vector3d(1,1,1), rgb);
}

// ========================================================================== //
// RGB <-> grayscale.
TYPED_TEST_P(TestConvertColorConversion, test_rgb_to_gray)
{
  typedef TypeParam T;
  // Using the explicit function.
  {
    Matrix<T, 3, 1> rgb(1,1,1);
    T gray;
    rgb_to_gray(rgb, gray);
    EXPECT_NEAR(gray, 1, T(1e-3));
  }
  // Using the unified API.
  {
    Pixel<T, Rgb> rgb(1,1,1);
    T gray;
    convert_color(rgb, gray);
    EXPECT_NEAR(gray, 1, T(1e-3));
  }
}

TYPED_TEST_P(TestConvertColorConversion, test_gray_to_rgb)
{
  // Using the explicit function.
  typedef TypeParam T;
  {
    const T src_gray = T(0.5);
    const Matrix<T, 3, 1> expected_rgb(src_gray, src_gray, src_gray);
    Matrix<T, 3, 1> actual_rgb;
    gray_to_rgb(src_gray, actual_rgb);
    EXPECT_EQ(expected_rgb, actual_rgb);
  }
  // Using the unified API.
  {
    const T src_gray = T(0.5);
    const Pixel<T, Rgb> expected_rgb(src_gray, src_gray, src_gray);
    Pixel<T, Rgb> actual_rgb;
    convert_color(src_gray, actual_rgb);
    EXPECT_EQ(expected_rgb, actual_rgb);
  }
}


// ========================================================================== //
// RGB <-> YUV.
TYPED_TEST_P(TestConvertColorConversion, test_rgb_to_yuv)
{
  typedef TypeParam T;
  typedef Matrix<T, 3, 1> Vec3;
  const Vec3 rgb[] = {
    Vec3(1, 0, 0),
    Vec3(0, 1, 0),
    Vec3(0, 0, 1)
  };
  const Vec3 expected_yuv[] = {
    Vec3(T(0.299), T(0.492*(0-0.299)), T(0.877*(1-0.299))),
    Vec3(T(0.587), T(0.492*(0-0.587)), T(0.877*(0-0.587))),
    Vec3(T(0.114), T(0.492*(1-0.114)), T(0.877*(0-0.114)))
  };
  // Using the explicit function.
  for (int i = 0; i < 3; ++i)
  {
    Vec3 actual_yuv;
    rgb_to_yuv(rgb[i], actual_yuv);
    EXPECT_MATRIX_NEAR(expected_yuv[i], actual_yuv, T(1e-3));
  }
  // Using the unified API.
  for (int i = 0; i < 3; ++i)
  {
    const Pixel<T, Rgb> rgb_pixel(rgb[i]);
    const Pixel<T, Yuv> expected_yuv_pixel(expected_yuv[i]);
    Pixel<T, Yuv> actual_yuv_pixel;
    convert_color(rgb_pixel, actual_yuv_pixel);
    EXPECT_MATRIX_NEAR(expected_yuv_pixel, actual_yuv_pixel, T(1e-3));
  }
}

TYPED_TEST_P(TestConvertColorConversion, test_yuv_to_rgb)
{
  typedef TypeParam T;
  typedef Matrix<T, 3, 1> Vec3;
  const Vec3 yuv[] = {
    Vec3(T(1), T(0)    , T(0)    ),
    Vec3(T(0), T(0.436), T(0)    ),
    Vec3(T(0), T(0)    , T(0.615))
  };
  const Vec3 expected_rgb[] = {
    Vec3(T(1)            , T(1)             , T(1)            ),
    Vec3(T(0)            , T(-0.39465*0.436), T(2.03211*0.436)),
    Vec3(T(1.13983*0.615), T(-0.58060*0.615), T(0)            )
  };
  // Using the explicit function.
  for (int i = 0; i < 3; ++i)
  {
    Vec3 actual_rgb;
    yuv_to_rgb(yuv[i], actual_rgb);
    EXPECT_MATRIX_NEAR(expected_rgb[i], actual_rgb, T(1e-3));
  }
  // Using the unified API.
  for (int i = 0; i < 3; ++i)
  {
    const Pixel<T, Yuv> yuv_pixel(yuv[i]);
    const Pixel<T, Yuv> expected_rgb_pixel(expected_rgb[i]);
    Pixel<T, Rgb> actual_rgb_pixel;
    convert_color(yuv_pixel, actual_rgb_pixel);
    EXPECT_MATRIX_NEAR(expected_rgb_pixel, actual_rgb_pixel, T(1e-3));
  }
}


// ========================================================================== //
// YUV <-> Gray
TYPED_TEST_P(TestConvertColorConversion, test_yuv_to_gray)
{
  typedef TypeParam T;
  typedef Matrix<T, 3, 1> Vec3;
  const Vec3 yuv(1, 0, 0);
  {
    T gray;
    yuv_to_gray(yuv, gray);
    EXPECT_EQ(1, gray);
  }
  {
    Pixel<T, Yuv> yuv_pixel(yuv);
    T gray;
    convert_color<T, Yuv>(yuv, gray);
    EXPECT_EQ(1, gray);
  }
}

TYPED_TEST_P(TestConvertColorConversion, test_gray_to_yuv)
{
  typedef TypeParam T;
  typedef Matrix<T, 3, 1> Vec3;
  T gray = T(0.5);
  {
    Vec3 yuv;
    const Vec3 expected_yuv(0.5, 0, 0);
    gray_to_yuv(gray, yuv);
    EXPECT_EQ(expected_yuv, yuv);
  }
  {
    Pixel<T, Yuv> yuv;
    const Pixel<T, Yuv> expected_yuv(0.5, 0, 0);
    convert_color(gray, yuv);
    EXPECT_EQ(expected_yuv, yuv);
  }
}


// ========================================================================== //
// Register all typed tests and instantiate them.
REGISTER_TYPED_TEST_CASE_P(TestConvertColorConversion,
                           test_rgb_to_gray,
                           test_gray_to_rgb,
                           test_rgb_to_yuv,
                           test_yuv_to_rgb,
                           test_yuv_to_gray,
                           test_gray_to_yuv);
INSTANTIATE_TYPED_TEST_CASE_P(Core_Pixel_ColorConversion,
                              TestConvertColorConversion,
                              FloatingPointChannelTypes);


// ========================================================================== //
// Run the tests.
int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}