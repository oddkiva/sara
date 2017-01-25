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

#include <limits>
#include <stdint.h>

#include <gtest/gtest.h>

#include <DO/Sara/Core/Pixel/SmartColorConversion.hpp>
#include <DO/Sara/Core/Pixel/Typedefs.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestSmartConvertColor, test_rgb_to_gray)
{
  const auto int16_max = numeric_limits<int16_t>::max();
  const auto uint8_max = numeric_limits<uint8_t>::max();

  auto rgbd = Pixel<double, Rgb>{ 1., 1., 1. };
  auto rgbf = Pixel<float, Rgb>{ 1.f, 1.f, 1.f };
  auto rgb16 = Pixel<int16_t, Rgb>{}; rgb16.fill(int16_max);
  auto rgb8u = Pixel<uint8_t, Rgb>{}; rgb8u.fill(uint8_max);

  auto grayd = double{};
  auto grayf = float{};
  auto gray8u = uint8_t{};
  auto gray16 = int16_t{};

  smart_convert_color(rgbd, grayd);  EXPECT_NEAR(grayd, 1, 1e-3);
  smart_convert_color(rgbd, grayf);  EXPECT_NEAR(grayf, 1, 1e-3);
  smart_convert_color(rgbd, gray16); EXPECT_EQ(gray16, int16_max);
  smart_convert_color(rgbd, gray8u); EXPECT_EQ(gray8u, uint8_max);

  smart_convert_color(rgbf, grayd);  EXPECT_NEAR(grayd, 1, 1e-3);
  smart_convert_color(rgbf, grayf);  EXPECT_NEAR(grayf, 1, 1e-3);
  smart_convert_color(rgbf, gray16); EXPECT_EQ(gray16, int16_max);
  smart_convert_color(rgbf, gray8u); EXPECT_EQ(gray8u, uint8_max);

  smart_convert_color(rgb16, grayd);  EXPECT_NEAR(grayd, 1, 1e-3);
  smart_convert_color(rgb16, grayf);  EXPECT_NEAR(grayf, 1, 1e-3);
  smart_convert_color(rgb16, gray16); EXPECT_EQ(gray16, int16_max);
  smart_convert_color(rgb16, gray8u); EXPECT_EQ(gray8u, uint8_max);

  smart_convert_color(rgb8u, grayd);  EXPECT_NEAR(grayd, 1, 1e-3);
  smart_convert_color(rgb8u, grayf);  EXPECT_NEAR(grayf, 1, 1e-3);
  smart_convert_color(rgb8u, gray16); EXPECT_EQ(gray16, int16_max);
  smart_convert_color(rgb8u, gray8u); EXPECT_EQ(gray8u, uint8_max);
}


TEST(TestSmartConvertColor, test_gray_to_rgb)
{
  const auto int16_max = numeric_limits<int16_t>::max();
  const auto uint8_max = numeric_limits<uint8_t>::max();

  auto grayd = 1.;
  auto grayf = 1.f;
  auto gray8u = uint8_max;
  auto gray16 = int16_max;

  auto rgbd = Pixel<double, Rgb>{};
  auto rgbf = Pixel<float, Rgb>{};
  auto rgb16 = Pixel<int16_t, Rgb>{};
  auto rgb8u = Pixel<uint8_t, Rgb>{};

  auto true_rgbd = Pixel<double, Rgb>{ Vector3d::Ones() };
  auto true_rgbf = Pixel<float, Rgb>{ Vector3f::Ones() };
  auto true_rgb16 = Pixel<int16_t, Rgb>{}; true_rgb16.fill(int16_max);
  auto true_rgb8u = Pixel<uint8_t, Rgb>{}; true_rgb8u.fill(uint8_max);

  smart_convert_color(grayd, rgbd);  EXPECT_MATRIX_NEAR(rgbd, true_rgbd, 1e-3);
  smart_convert_color(grayd, rgbf);  EXPECT_MATRIX_NEAR(rgbf, true_rgbf, 1e-3);
  smart_convert_color(grayd, rgb16); EXPECT_MATRIX_EQ(rgb16, true_rgb16);
  smart_convert_color(grayd, rgb8u); EXPECT_MATRIX_EQ(rgb8u, true_rgb8u);

  smart_convert_color(grayf, rgbd);  EXPECT_MATRIX_NEAR(rgbd, true_rgbd, 1e-3);
  smart_convert_color(grayf, rgbf);  EXPECT_MATRIX_NEAR(rgbf, true_rgbf, 1e-3);
  smart_convert_color(grayf, rgb16); EXPECT_MATRIX_EQ(rgb16, true_rgb16);
  smart_convert_color(grayf, rgb8u); EXPECT_MATRIX_EQ(rgb8u, true_rgb8u);

  smart_convert_color(gray16, rgbd);  EXPECT_MATRIX_NEAR(rgbd, true_rgbd, 1e-3);
  smart_convert_color(gray16, rgbf);  EXPECT_MATRIX_NEAR(rgbf, true_rgbf, 1e-3);
  smart_convert_color(gray16, rgb16); EXPECT_MATRIX_EQ(rgb16, true_rgb16);
  smart_convert_color(gray16, rgb8u); EXPECT_MATRIX_EQ(rgb8u, true_rgb8u);

  smart_convert_color(gray8u, rgbd);  EXPECT_MATRIX_NEAR(rgbd, true_rgbd, 1e-3);
  smart_convert_color(gray8u, rgbf);  EXPECT_MATRIX_NEAR(rgbf, true_rgbf, 1e-3);
  smart_convert_color(gray8u, rgb16); EXPECT_MATRIX_EQ(rgb16, true_rgb16);
  smart_convert_color(gray8u, rgb8u); EXPECT_MATRIX_EQ(rgb8u, true_rgb8u);
}


TEST(TestSmartConvertColor, test_rgb_to_yuv)
{
  const auto int16_max = numeric_limits<int16_t>::max();
  const auto int16_min = numeric_limits<int16_t>::min();
  const auto uint8_max = numeric_limits<uint8_t>::max();

  auto yuvd = Pixel<double, Yuv>{ 1., 0., 0. };
  auto yuvf = Pixel<float, Yuv>{ 1.f, 0.f, 0.f };
  auto yuv16 = Pixel<int16_t, Yuv>{ int16_max, int16_min, int16_min };
  auto yuv8u = Pixel<uint8_t, Yuv>{ uint8_max, 0, 0 };

  auto rgbd = Pixel<double, Rgb>{};
  auto rgbf = Pixel<float, Rgb>{};
  auto rgb16 = Pixel<int16_t, Rgb>{};
  auto rgb8u = Pixel<uint8_t, Rgb>{};

  auto true_rgbd = Pixel<double, Rgb>{ 1., 1., 1. };
  auto true_rgbf = Pixel<float, Rgb>{ 1.f, 1.f, 1.f };
  auto true_rgb16 = Pixel<int16_t, Rgb>{}; true_rgb16.fill(int16_max);
  auto true_rgb8u = Pixel<uint8_t, Rgb>{}; true_rgb8u.fill(uint8_max);

  smart_convert_color(yuvd, rgbd);  EXPECT_MATRIX_NEAR(rgbd, true_rgbd, 1e-3);
  smart_convert_color(yuvd, rgbf);  EXPECT_MATRIX_NEAR(rgbf, true_rgbf, 1e-3);
  smart_convert_color(yuvd, rgb16); EXPECT_MATRIX_EQ(rgb16, true_rgb16);
  smart_convert_color(yuvd, rgb8u); EXPECT_MATRIX_EQ(rgb8u, true_rgb8u);

  smart_convert_color(yuvf, rgbd);  EXPECT_MATRIX_NEAR(rgbd, true_rgbd, 1e-3);
  smart_convert_color(yuvf, rgbf);  EXPECT_MATRIX_NEAR(rgbf, true_rgbf, 1e-3);
  smart_convert_color(yuvf, rgb16); EXPECT_MATRIX_EQ(rgb16, true_rgb16);
  smart_convert_color(yuvf, rgb8u); EXPECT_MATRIX_EQ(rgb8u, true_rgb8u);

  smart_convert_color(yuv16, rgbd);  EXPECT_MATRIX_NEAR(rgbd, true_rgbd, 1e-3);
  smart_convert_color(yuv16, rgbf);  EXPECT_MATRIX_NEAR(rgbf, true_rgbf, 1e-3);
  smart_convert_color(yuv16, rgb16); EXPECT_MATRIX_EQ(rgb16, true_rgb16);
  smart_convert_color(yuv16, rgb8u); EXPECT_MATRIX_EQ(rgb8u, true_rgb8u);

  smart_convert_color(yuv8u, rgbd);  EXPECT_MATRIX_NEAR(rgbd, true_rgbd, 1e-3);
  smart_convert_color(yuv8u, rgbf);  EXPECT_MATRIX_NEAR(rgbf, true_rgbf, 1e-3);
  smart_convert_color(yuv8u, rgb16); EXPECT_MATRIX_EQ(rgb16, true_rgb16);
  smart_convert_color(yuv8u, rgb8u); EXPECT_MATRIX_EQ(rgb8u, true_rgb8u);
}


TEST(TestSmartConvertColor, test_gray_to_gray)
{
  const auto int16_max = numeric_limits<int16_t>::max();
  const auto uint8_max = numeric_limits<uint8_t>::max();

  auto grayd = 1.;
  auto grayf = 1.f;
  auto gray8u = uint8_max;
  auto gray16 = int16_max;

  auto dst_grayd = double{};
  auto dst_grayf = float{};
  auto dst_gray8u = uint8_t{};
  auto dst_gray16 = int16_t{};

  smart_convert_color(grayd, dst_grayf);  EXPECT_NEAR(dst_grayf, 1, 1e-3);
  smart_convert_color(grayd, dst_gray16); EXPECT_EQ(dst_gray16, int16_max);
  smart_convert_color(grayd, dst_gray8u); EXPECT_EQ(dst_gray8u, uint8_max);

  smart_convert_color(grayf, dst_grayd);  EXPECT_NEAR(dst_grayd, 1, 1e-3);
  smart_convert_color(grayf, dst_gray16); EXPECT_EQ(dst_gray16, int16_max);
  smart_convert_color(grayf, dst_gray8u); EXPECT_EQ(dst_gray8u, uint8_max);

  smart_convert_color(gray16, dst_grayd);  EXPECT_NEAR(dst_grayd, 1, 1e-3);
  smart_convert_color(gray16, dst_grayf);  EXPECT_NEAR(dst_grayf, 1, 1e-3);
  smart_convert_color(gray16, dst_gray8u); EXPECT_EQ(dst_gray8u, uint8_max);

  smart_convert_color(gray8u, dst_grayd);  EXPECT_NEAR(dst_grayd, 1, 1e-3);
  smart_convert_color(gray8u, dst_grayf);  EXPECT_NEAR(dst_grayf, 1, 1e-3);
  smart_convert_color(gray8u, dst_gray16); EXPECT_EQ(dst_gray16, int16_max);
}


TEST(TestSmartConvertColor, test_corner_cases)
{
  {
    auto src = Rgb8{ 0, 0, 255 };
    auto dst = Rgb32f{};

    smart_convert_color(src, dst);
    EXPECT_MATRIX_EQ(Rgb32f(0, 0, 1), dst);
  }

  {
    auto src = Rgb32f{ 0.f, 0.f, 1.f };
    auto dst = Rgb64f{};

    smart_convert_color(src, dst);
    EXPECT_MATRIX_EQ(Rgb64f(0, 0, 1), dst);
  }

  {
    auto src = Rgb64f{ 0., 0., 1. };
    auto dst = Rgb32f{};

    smart_convert_color(src, dst);
    EXPECT_MATRIX_EQ(Rgb32f(0, 0, 1), dst);
  }
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
