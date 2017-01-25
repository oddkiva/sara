// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <gtest/gtest.h>

#include <DO/Sara/Core/Pixel/Pixel.hpp>
#include <DO/Sara/Core/Pixel/ColorConversion.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


// ========================================================================== //
// Define the set of integral channel types, which we will test.
using FloatingPointChannelTypes = testing::Types<float, double>;


// ========================================================================== //
// Define the parameterized test case.
template <typename ChannelType>
class TestConvertColor : public testing::Test {};
TYPED_TEST_CASE_P(TestConvertColor);


// ========================================================================== //
// RGB <-> RGBA.
TEST(TestConvertColor, test_rgb_to_rgba)
{
  Pixel<double, Rgb> rgb(1,1,1);
  Pixel<double, Rgba> rgba;
  convert_color(rgb, rgba);
  EXPECT_EQ(Vector4d(1,1,1,1), rgba);
}

TEST(TestConvertColor, test_rgba_to_rgb)
{
  Pixel<double, Rgba> rgba(1,1,1,1);
  Pixel<double, Rgb> rgb;
  convert_color(rgba, rgb);
  EXPECT_EQ(Vector3d(1,1,1), rgb);
}

// ========================================================================== //
// RGB <-> grayscale.
TYPED_TEST_P(TestConvertColor, test_rgb_to_gray)
{
  using T = TypeParam;

  // Using the explicit function.
  {
    auto rgb = Matrix<T, 3, 1>(1,1,1);
    auto gray = T{};
    rgb_to_gray(rgb, gray);
    EXPECT_NEAR(gray, 1, T(1e-3));
  }
  // Using the unified API.
  {
    auto rgb = Pixel<T, Rgb>(1,1,1);
    auto gray = T{};
    convert_color(rgb, gray);
    EXPECT_NEAR(gray, 1, T(1e-3));
  }
}

TYPED_TEST_P(TestConvertColor, test_gray_to_rgb)
{
  // Using the explicit function.
  using T = TypeParam;
  {
    const auto src_gray = T(0.5);
    const auto expected_rgb = Matrix<T, 3, 1>(src_gray, src_gray, src_gray);
    auto actual_rgb = Matrix<T, 3, 1>{};
    gray_to_rgb(src_gray, actual_rgb);
    EXPECT_EQ(expected_rgb, actual_rgb);
  }
  // Using the unified API.
  {
    const auto src_gray = T(0.5);
    const auto expected_rgb = Pixel<T, Rgb>(src_gray, src_gray, src_gray);
    auto actual_rgb = Pixel<T, Rgb>{};
    convert_color(src_gray, actual_rgb);
    EXPECT_EQ(expected_rgb, actual_rgb);
  }
}


// ========================================================================== //
// RGB <-> YUV.
TYPED_TEST_P(TestConvertColor, test_rgb_to_yuv)
{
  using T = TypeParam;
  using Vec3 = Matrix<T, 3, 1>;

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
    auto actual_yuv = Vec3{};
    rgb_to_yuv(rgb[i], actual_yuv);
    EXPECT_MATRIX_NEAR(expected_yuv[i], actual_yuv, T(1e-3));
  }
  // Using the unified API.
  for (int i = 0; i < 3; ++i)
  {
    const auto rgb_pixel = Pixel<T, Rgb>{ rgb[i] };
    const auto expected_yuv_pixel = Pixel<T, Yuv>{ expected_yuv[i] };
    auto actual_yuv_pixel = Pixel<T, Yuv>{};
    convert_color(rgb_pixel, actual_yuv_pixel);
    EXPECT_MATRIX_NEAR(expected_yuv_pixel, actual_yuv_pixel, T(1e-3));
  }
}

TYPED_TEST_P(TestConvertColor, test_yuv_to_rgb)
{
  using T = TypeParam;
  using Vec3 = Matrix<T, 3, 1>;

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
    auto actual_rgb = Vec3{};
    yuv_to_rgb(yuv[i], actual_rgb);
    EXPECT_MATRIX_NEAR(expected_rgb[i], actual_rgb, T(1e-3));
  }
  // Using the unified API.
  for (int i = 0; i < 3; ++i)
  {
    const auto yuv_pixel = Pixel<T, Yuv>{ yuv[i] };
    const auto expected_rgb_pixel = Pixel<T, Yuv>{ expected_rgb[i] };
    auto actual_rgb_pixel = Pixel<T, Rgb>{};
    convert_color(yuv_pixel, actual_rgb_pixel);
    EXPECT_MATRIX_NEAR(expected_rgb_pixel, actual_rgb_pixel, T(1e-3));
  }
}


// ========================================================================== //
// YUV <-> Gray
TYPED_TEST_P(TestConvertColor, test_yuv_to_gray)
{
  using T = TypeParam;
  using Vec3 = Matrix<T, 3, 1>;

  const auto yuv = Vec3(1, 0, 0);
  {
    auto gray = T{};
    yuv_to_gray(yuv, gray);
    EXPECT_EQ(1, gray);
  }
  {
    auto yuv_pixel = Pixel<T, Yuv>{ yuv };
    auto gray = T{};
    convert_color<T, Yuv>(yuv_pixel, gray);
    EXPECT_EQ(1, gray);
  }
}

TYPED_TEST_P(TestConvertColor, test_gray_to_yuv)
{
  using T = TypeParam;
  using Vec3 = Matrix<T, 3, 1>;

  auto gray = T(0.5);
  {
    auto yuv = Vec3{};
    const auto expected_yuv = Vec3(0.5, 0, 0);
    gray_to_yuv(gray, yuv);
    EXPECT_EQ(expected_yuv, yuv);
  }
  {
    auto yuv = Pixel<T, Yuv>{};
    const auto expected_yuv = Pixel<T, Yuv>(0.5, 0, 0);
    convert_color(gray, yuv);
    EXPECT_EQ(expected_yuv, yuv);
  }
}


// ========================================================================== //
// Register all typed tests and instantiate them.
REGISTER_TYPED_TEST_CASE_P(TestConvertColor,
                           test_rgb_to_gray,
                           test_gray_to_rgb,
                           test_rgb_to_yuv,
                           test_yuv_to_rgb,
                           test_yuv_to_gray,
                           test_gray_to_yuv);
INSTANTIATE_TYPED_TEST_CASE_P(Core_Pixel_ColorConversion,
                              TestConvertColor,
                              FloatingPointChannelTypes);


// ========================================================================== //
// Run the tests.
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
