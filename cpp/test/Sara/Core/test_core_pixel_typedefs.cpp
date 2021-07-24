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

#define BOOST_TEST_MODULE "Core/Pixel/Pixel Aliases"

#include <cstdint>

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Meta.hpp>
#include <DO/Sara/Core/Pixel/Typedefs.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestPixelTypedefs)

BOOST_AUTO_TEST_CASE(test_3d_colors_typedefs)
{
  // RgbXX with unsigned integer types.
  static_assert(std::is_same<Rgb8::base_type, Matrix<unsigned char, 3, 1>>::value, "");
  static_assert(
      std::is_same<Rgb16::base_type, Matrix<unsigned short, 3, 1>>::value, "");
  static_assert(
      std::is_same<Rgb32::base_type, Matrix<unsigned int, 3, 1>>::value, "");

  // RgbX with signed integer types.
  static_assert(std::is_same<Rgb8s::base_type, Matrix<char, 3, 1>>::value, "");
  static_assert(std::is_same<Rgb16s::base_type, Matrix<short, 3, 1>>::value,
                "");
  static_assert(std::is_same<Rgb32s::base_type, Matrix<int, 3, 1>>::value, "");

  // RgbX with floating-point types.
  static_assert(std::is_same<Rgb32f::base_type, Matrix<float, 3, 1>>::value, "");
  static_assert(std::is_same<Rgb64f::base_type, Matrix<double, 3, 1>>::value, "");
}

BOOST_AUTO_TEST_CASE(test_4d_colors_typedefs)
{
  // RgbaXX with unsigned integer types.
  static_assert(std::is_same<Rgba8::base_type, Matrix<unsigned char, 4, 1>>::value, "");
  static_assert(std::is_same<Rgba16::base_type, Matrix<unsigned short, 4, 1>>::value,
                "");
  static_assert(std::is_same<Rgba32::base_type, Matrix<unsigned int, 4, 1>>::value, "");

  // RgbaX with signed integer types.
  static_assert(std::is_same<Rgba8s::base_type, Matrix<char, 4, 1>>::value, "");
  static_assert(std::is_same<Rgba16s::base_type, Matrix<short, 4, 1>>::value, "");
  static_assert(std::is_same<Rgba32s::base_type, Matrix<int, 4, 1>>::value, "");

  // RgbaX with floating-point types.
  static_assert(std::is_same<Rgba32f::base_type, Matrix<float, 4, 1>>::value, "");
  static_assert(std::is_same<Rgba64f::base_type, Matrix<double, 4, 1>>::value, "");
}

BOOST_AUTO_TEST_CASE(test_rgb_color_constants)
{
  // Check colors with signed char channels.
  BOOST_CHECK_EQUAL(Rgb8s(127, -128, -128), red<char>());
  BOOST_CHECK_EQUAL(Rgb8s(-128, 127, -128), green<char>());
  BOOST_CHECK_EQUAL(Rgb8s(-128, -128, 127), blue<char>());
  BOOST_CHECK_EQUAL(Rgb8s(-128, 127, 127), cyan<char>());
  BOOST_CHECK_EQUAL(Rgb8s(127, -128, 127), magenta<char>());
  BOOST_CHECK_EQUAL(Rgb8s(127, 127, -128), yellow<char>());
  BOOST_CHECK_EQUAL(Rgb8s(-128, -128, -128), black<char>());

  // Check colors with unsigned char channels.
  BOOST_CHECK_EQUAL(Rgb8(255, 0, 0), Red8);
  BOOST_CHECK_EQUAL(Rgb8(0, 255, 0), Green8);
  BOOST_CHECK_EQUAL(Rgb8(0, 0, 255), Blue8);
  BOOST_CHECK_EQUAL(Rgb8(0, 255, 255), Cyan8);
  BOOST_CHECK_EQUAL(Rgb8(255, 0, 255), Magenta8);
  BOOST_CHECK_EQUAL(Rgb8(255, 255, 0), Yellow8);
  BOOST_CHECK_EQUAL(Rgb8(0, 0, 0), Black8);

  // Check colors
  BOOST_CHECK_EQUAL(Rgb32f(1, 0, 0), red<float>());
  BOOST_CHECK_EQUAL(Rgb32f(0, 1, 0), green<float>());
  BOOST_CHECK_EQUAL(Rgb32f(0, 0, 1), blue<float>());
  BOOST_CHECK_EQUAL(Rgb32f(0, 1, 1), cyan<float>());
  BOOST_CHECK_EQUAL(Rgb32f(1, 0, 1), magenta<float>());
  BOOST_CHECK_EQUAL(Rgb32f(1, 1, 0), yellow<float>());
  BOOST_CHECK_EQUAL(Rgb32f(0, 0, 0), black<float>());
}

BOOST_AUTO_TEST_SUITE_END()
