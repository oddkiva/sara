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

#define BOOST_TEST_MODULE "Core/Pixel/Smart Color Conversions"

#include <cstdint>
#include <limits>

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Pixel/SmartColorConversion.hpp>
#include <DO/Sara/Core/Pixel/Typedefs.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestSmartConvertColor)

BOOST_AUTO_TEST_CASE(test_rgb_to_gray)
{
  const auto int16_max = numeric_limits<int16_t>::max();
  const auto uint8_max = numeric_limits<uint8_t>::max();

  auto rgbd = Pixel<double, Rgb>{1., 1., 1.};
  auto rgbf = Pixel<float, Rgb>{1.f, 1.f, 1.f};
  auto rgb16 = Pixel<int16_t, Rgb>{};
  rgb16.fill(int16_max);
  auto rgb8u = Pixel<uint8_t, Rgb>{};
  rgb8u.fill(uint8_max);

  auto grayd = double{};
  auto grayf = float{};
  auto gray8u = uint8_t{};
  auto gray16 = int16_t{};

  smart_convert_color(rgbd, grayd);
  BOOST_CHECK_CLOSE(grayd, 1, 1e-3);
  smart_convert_color(rgbd, grayf);
  BOOST_CHECK_CLOSE(grayf, 1, 1e-3);
  smart_convert_color(rgbd, gray16);
  BOOST_CHECK_EQUAL(gray16, int16_max);
  smart_convert_color(rgbd, gray8u);
  BOOST_CHECK_EQUAL(gray8u, uint8_max);

  smart_convert_color(rgbf, grayd);
  BOOST_CHECK_CLOSE(grayd, 1, 1e-3);
  smart_convert_color(rgbf, grayf);
  BOOST_CHECK_CLOSE(grayf, 1, 1e-3);
  smart_convert_color(rgbf, gray16);
  BOOST_CHECK_EQUAL(gray16, int16_max);
  smart_convert_color(rgbf, gray8u);
  BOOST_CHECK_EQUAL(gray8u, uint8_max);

  smart_convert_color(rgb16, grayd);
  BOOST_CHECK_CLOSE(grayd, 1, 1e-3);
  smart_convert_color(rgb16, grayf);
  BOOST_CHECK_CLOSE(grayf, 1, 1e-3);
  smart_convert_color(rgb16, gray16);
  BOOST_CHECK_EQUAL(gray16, int16_max);
  smart_convert_color(rgb16, gray8u);
  BOOST_CHECK_EQUAL(gray8u, uint8_max);

  smart_convert_color(rgb8u, grayd);
  BOOST_CHECK_CLOSE(grayd, 1, 1e-3);
  smart_convert_color(rgb8u, grayf);
  BOOST_CHECK_CLOSE(grayf, 1, 1e-3);
  smart_convert_color(rgb8u, gray16);
  BOOST_CHECK_EQUAL(gray16, int16_max);
  smart_convert_color(rgb8u, gray8u);
  BOOST_CHECK_EQUAL(gray8u, uint8_max);
}


BOOST_AUTO_TEST_CASE(test_gray_to_rgb)
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

  auto true_rgbd = Pixel<double, Rgb>{Vector3d::Ones()};
  auto true_rgbf = Pixel<float, Rgb>{Vector3f::Ones()};
  auto true_rgb16 = Pixel<int16_t, Rgb>{};
  true_rgb16.fill(int16_max);
  auto true_rgb8u = Pixel<uint8_t, Rgb>{};
  true_rgb8u.fill(uint8_max);

  smart_convert_color(grayd, rgbd);
  BOOST_CHECK_SMALL((rgbd - true_rgbd).norm(), 1e-3);
  smart_convert_color(grayd, rgbf);
  BOOST_CHECK_SMALL((rgbf - true_rgbf).norm(), 1e-3f);
  smart_convert_color(grayd, rgb16);
  BOOST_CHECK_EQUAL(rgb16, true_rgb16);
  smart_convert_color(grayd, rgb8u);
  BOOST_CHECK_EQUAL(rgb8u, true_rgb8u);

  smart_convert_color(grayf, rgbd);
  BOOST_CHECK_SMALL((rgbd - true_rgbd).norm(), 1e-3);
  smart_convert_color(grayf, rgbf);
  BOOST_CHECK_SMALL((rgbf - true_rgbf).norm(), 1e-3f);
  smart_convert_color(grayf, rgb16);
  BOOST_CHECK_EQUAL(rgb16, true_rgb16);
  smart_convert_color(grayf, rgb8u);
  BOOST_CHECK_EQUAL(rgb8u, true_rgb8u);

  smart_convert_color(gray16, rgbd);
  BOOST_CHECK_SMALL((rgbd - true_rgbd).norm(), 1e-3);
  smart_convert_color(gray16, rgbf);
  BOOST_CHECK_SMALL((rgbf - true_rgbf).norm(), 1e-3f);
  smart_convert_color(gray16, rgb16);
  BOOST_CHECK_EQUAL(rgb16, true_rgb16);
  smart_convert_color(gray16, rgb8u);
  BOOST_CHECK_EQUAL(rgb8u, true_rgb8u);

  smart_convert_color(gray8u, rgbd);
  BOOST_CHECK_SMALL((rgbd - true_rgbd).norm(), 1e-3);
  smart_convert_color(gray8u, rgbf);
  BOOST_CHECK_SMALL((rgbf - true_rgbf).norm(), 1e-3f);
  smart_convert_color(gray8u, rgb16);
  BOOST_CHECK_EQUAL(rgb16, true_rgb16);
  smart_convert_color(gray8u, rgb8u);
  BOOST_CHECK_EQUAL(rgb8u, true_rgb8u);
}


BOOST_AUTO_TEST_CASE(test_rgb_to_yuv)
{
  const auto int16_max = numeric_limits<int16_t>::max();
  const auto int16_min = numeric_limits<int16_t>::min();
  const auto uint8_max = numeric_limits<uint8_t>::max();

  auto yuvd = Pixel<double, Yuv>{1., 0., 0.};
  auto yuvf = Pixel<float, Yuv>{1.f, 0.f, 0.f};
  auto yuv16 = Pixel<int16_t, Yuv>{int16_max, int16_min, int16_min};
  auto yuv8u = Pixel<uint8_t, Yuv>{uint8_max, 0, 0};

  auto rgbd = Pixel<double, Rgb>{};
  auto rgbf = Pixel<float, Rgb>{};
  auto rgb16 = Pixel<int16_t, Rgb>{};
  auto rgb8u = Pixel<uint8_t, Rgb>{};

  auto true_rgbd = Pixel<double, Rgb>{1., 1., 1.};
  auto true_rgbf = Pixel<float, Rgb>{1.f, 1.f, 1.f};
  auto true_rgb16 = Pixel<int16_t, Rgb>{};
  true_rgb16.fill(int16_max);
  auto true_rgb8u = Pixel<uint8_t, Rgb>{};
  true_rgb8u.fill(uint8_max);

  smart_convert_color(yuvd, rgbd);
  BOOST_CHECK_SMALL((rgbd - true_rgbd).norm(), 1e-3);
  smart_convert_color(yuvd, rgbf);
  BOOST_CHECK_SMALL((rgbf - true_rgbf).norm(), 1e-3f);
  smart_convert_color(yuvd, rgb16);
  BOOST_CHECK_EQUAL(rgb16, true_rgb16);
  smart_convert_color(yuvd, rgb8u);
  BOOST_CHECK_EQUAL(rgb8u, true_rgb8u);

  smart_convert_color(yuvf, rgbd);
  BOOST_CHECK_SMALL((rgbd - true_rgbd).norm(), 1e-3);
  smart_convert_color(yuvf, rgbf);
  BOOST_CHECK_SMALL((rgbf - true_rgbf).norm(), 1e-3f);
  smart_convert_color(yuvf, rgb16);
  BOOST_CHECK_EQUAL(rgb16, true_rgb16);
  smart_convert_color(yuvf, rgb8u);
  BOOST_CHECK_EQUAL(rgb8u, true_rgb8u);

  smart_convert_color(yuv16, rgbd);
  BOOST_CHECK_SMALL((rgbd - true_rgbd).norm(), 1e-3);
  smart_convert_color(yuv16, rgbf);
  BOOST_CHECK_SMALL((rgbf - true_rgbf).norm(), 1e-3f);
  smart_convert_color(yuv16, rgb16);
  BOOST_CHECK_EQUAL(rgb16, true_rgb16);
  smart_convert_color(yuv16, rgb8u);
  BOOST_CHECK_EQUAL(rgb8u, true_rgb8u);

  smart_convert_color(yuv8u, rgbd);
  BOOST_CHECK_SMALL((rgbd - true_rgbd).norm(), 1e-3);
  smart_convert_color(yuv8u, rgbf);
  BOOST_CHECK_SMALL((rgbf - true_rgbf).norm(), 1e-3f);
  smart_convert_color(yuv8u, rgb16);
  BOOST_CHECK_EQUAL(rgb16, true_rgb16);
  smart_convert_color(yuv8u, rgb8u);
  BOOST_CHECK_EQUAL(rgb8u, true_rgb8u);
}


BOOST_AUTO_TEST_CASE(test_gray_to_gray)
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

  smart_convert_color(grayd, dst_grayf);
  BOOST_CHECK_CLOSE(dst_grayf, 1, 1e-3);
  smart_convert_color(grayd, dst_gray16);
  BOOST_CHECK_EQUAL(dst_gray16, int16_max);
  smart_convert_color(grayd, dst_gray8u);
  BOOST_CHECK_EQUAL(dst_gray8u, uint8_max);

  smart_convert_color(grayf, dst_grayd);
  BOOST_CHECK_CLOSE(dst_grayd, 1, 1e-3);
  smart_convert_color(grayf, dst_gray16);
  BOOST_CHECK_EQUAL(dst_gray16, int16_max);
  smart_convert_color(grayf, dst_gray8u);
  BOOST_CHECK_EQUAL(dst_gray8u, uint8_max);

  smart_convert_color(gray16, dst_grayd);
  BOOST_CHECK_CLOSE(dst_grayd, 1, 1e-3);
  smart_convert_color(gray16, dst_grayf);
  BOOST_CHECK_CLOSE(dst_grayf, 1, 1e-3);
  smart_convert_color(gray16, dst_gray8u);
  BOOST_CHECK_EQUAL(dst_gray8u, uint8_max);

  smart_convert_color(gray8u, dst_grayd);
  BOOST_CHECK_CLOSE(dst_grayd, 1, 1e-3);
  smart_convert_color(gray8u, dst_grayf);
  BOOST_CHECK_CLOSE(dst_grayf, 1, 1e-3);
  smart_convert_color(gray8u, dst_gray16);
  BOOST_CHECK_EQUAL(dst_gray16, int16_max);
}


BOOST_AUTO_TEST_CASE(test_corner_cases)
{
  {
    auto src = Rgb8{0, 0, 255};
    auto dst = Rgb32f{};

    smart_convert_color(src, dst);
    BOOST_CHECK_EQUAL(Rgb32f(0, 0, 1), dst);
  }

  {
    auto src = Rgb32f{0.f, 0.f, 1.f};
    auto dst = Rgb64f{};

    smart_convert_color(src, dst);
    BOOST_CHECK_EQUAL(Rgb64f(0, 0, 1), dst);
  }

  {
    auto src = Rgb64f{0., 0., 1.};
    auto dst = Rgb32f{};

    smart_convert_color(src, dst);
    BOOST_CHECK_EQUAL(Rgb32f(0, 0, 1), dst);
  }
}

BOOST_AUTO_TEST_SUITE_END()
