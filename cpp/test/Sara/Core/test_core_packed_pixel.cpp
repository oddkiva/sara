// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Core/Pixel/Packed Pixel class"

#include <limits>
#include <cstdint>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Pixel/PackedPixel.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestPackedPixel)

BOOST_AUTO_TEST_CASE(test_initialization_for_3d_packed_pixel)
{
  using pixel_base_type = PackedPixelBase_3<uint16_t, 5, 6, 5>;
  BOOST_CHECK_EQUAL(sizeof(pixel_base_type), 2u);

  auto red = pixel_base_type{31, 0, 0};
  BOOST_CHECK_EQUAL(*reinterpret_cast<uint16_t*>(&red), 0x001f);
  BOOST_CHECK_EQUAL(red.channel_0, 31);
  BOOST_CHECK_EQUAL(red.channel_1, 0);
  BOOST_CHECK_EQUAL(red.channel_2, 0);

  auto green = pixel_base_type{0, 63, 0};
  BOOST_CHECK_EQUAL(*reinterpret_cast<uint16_t*>(&green), 0x07e0);
  BOOST_CHECK_EQUAL(green.channel_0, 0);
  BOOST_CHECK_EQUAL(green.channel_1, 63);
  BOOST_CHECK_EQUAL(green.channel_2, 0);

  auto blue = pixel_base_type{0, 0, 31};
  BOOST_CHECK_EQUAL(*reinterpret_cast<uint16_t*>(&blue), 0xf800);
  BOOST_CHECK_EQUAL(blue.channel_0, 0);
  BOOST_CHECK_EQUAL(blue.channel_1, 0);
  BOOST_CHECK_EQUAL(blue.channel_2, 31);

  auto white = pixel_base_type{31, 63, 31};
  BOOST_CHECK_EQUAL(*reinterpret_cast<uint16_t*>(&white), 0xffff);
  BOOST_CHECK_EQUAL(white.channel_0, 31);
  BOOST_CHECK_EQUAL(white.channel_1, 63);
  BOOST_CHECK_EQUAL(white.channel_2, 31);

  auto black = pixel_base_type{0, 0, 0};
  BOOST_CHECK_EQUAL(*reinterpret_cast<uint16_t*>(&black), 0x0);
  BOOST_CHECK_EQUAL(black.channel_0, 0);
  BOOST_CHECK_EQUAL(black.channel_1, 0);
  BOOST_CHECK_EQUAL(black.channel_2, 0);
}

BOOST_AUTO_TEST_CASE(test_initialization_for_4d_packed_pixel)
{
  using rgba_pixel_type = PackedPixelBase_4<uint8_t, 8, 8, 8, 8>;
  BOOST_CHECK_EQUAL(sizeof(rgba_pixel_type), 4u);

  auto red = rgba_pixel_type{255, 0, 0, 255};
  auto true_red = uint32_t{0xff0000ff};
  BOOST_CHECK(*reinterpret_cast<uint32_t*>(&red) == true_red);
  BOOST_CHECK_EQUAL(red.channel_0, 255);
  BOOST_CHECK_EQUAL(red.channel_1, 0);
  BOOST_CHECK_EQUAL(red.channel_2, 0);
  BOOST_CHECK_EQUAL(red.channel_3, 255);

  auto green = rgba_pixel_type{0, 255, 0, 255};
  auto true_green = uint32_t{0xff00ff00};
  BOOST_CHECK(*reinterpret_cast<uint32_t*>(&green) == true_green);
  BOOST_CHECK_EQUAL(green.channel_0, 0);
  BOOST_CHECK_EQUAL(green.channel_1, 255);
  BOOST_CHECK_EQUAL(green.channel_2, 0);
  BOOST_CHECK_EQUAL(green.channel_3, 255);

  auto blue = rgba_pixel_type{0, 0, 255, 255};
  auto true_blue = uint32_t{0xffff0000};
  BOOST_CHECK(*reinterpret_cast<uint32_t*>(&blue) == true_blue);
  BOOST_CHECK_EQUAL(blue.channel_0, 0);
  BOOST_CHECK_EQUAL(blue.channel_1, 0);
  BOOST_CHECK_EQUAL(blue.channel_2, 255);
  BOOST_CHECK_EQUAL(blue.channel_3, 255);

  auto black = rgba_pixel_type{0, 0, 0, 255};
  auto true_black = uint32_t{0xff000000};
  BOOST_CHECK(*reinterpret_cast<uint32_t*>(&black) == true_black);
  BOOST_CHECK_EQUAL(black.channel_0, 0);
  BOOST_CHECK_EQUAL(black.channel_1, 0);
  BOOST_CHECK_EQUAL(black.channel_2, 0);
  BOOST_CHECK_EQUAL(black.channel_3, 255);

  auto white = rgba_pixel_type{255, 255, 255, 255};
  auto true_white = uint32_t{0xffffffff};
  BOOST_CHECK(*reinterpret_cast<uint32_t*>(&white) == true_white);
  BOOST_CHECK_EQUAL(white.channel_0, 255);
  BOOST_CHECK_EQUAL(white.channel_1, 255);
  BOOST_CHECK_EQUAL(white.channel_2, 255);
  BOOST_CHECK_EQUAL(white.channel_3, 255);
}

BOOST_AUTO_TEST_CASE(test_channel_index)
{
  using BgrModel = ColorModel<Rgb, Meta::IntArray_3<2, 1, 0>>;
  const auto r = ChannelIndex<BgrModel, R>::value;
  const auto g = ChannelIndex<BgrModel, G>::value;
  const auto b = ChannelIndex<BgrModel, B>::value;
  BOOST_CHECK_EQUAL(r, 2);
  BOOST_CHECK_EQUAL(g, 1);
  BOOST_CHECK_EQUAL(b, 0);
}

BOOST_AUTO_TEST_CASE(test_arithmetic_operations)
{
  using Pixel_565 = PackedPixelBase_3<uint16_t, 5, 6, 5>;
  using BgrModel = ColorModel<Rgb, Meta::IntArray_3<2, 1, 0>>;
  using Bgr_565 = PackedPixel<Pixel_565, BgrModel>;

  // Constructor.
  Bgr_565 p1{1, 2, 0};
  BOOST_CHECK(p1.channel<B>() == 1);
  BOOST_CHECK(p1.channel<G>() == 2);
  BOOST_CHECK(p1.channel<R>() == 0);

  // Constructor.
  Bgr_565 p2{1, 3, 1};
  BOOST_CHECK(p2.channel<B>() == 1);
  BOOST_CHECK(p2.channel<G>() == 3);
  BOOST_CHECK(p2.channel<R>() == 1);

  // Assignment.
  p2 = p1;
  BOOST_CHECK(p2.channel<B>() == 1);
  BOOST_CHECK(p2.channel<G>() == 2);
  BOOST_CHECK(p2.channel<R>() == 0);

  // Copy constructor.
  Bgr_565 p{p1};
  BOOST_CHECK(p.channel<B>() == 1);
  BOOST_CHECK(p.channel<G>() == 2);
  BOOST_CHECK(p.channel<R>() == 0);

  // Addition.
  p = p1 + p2;
  BOOST_CHECK(p.channel<R>() == 0);
  BOOST_CHECK(p.channel<G>() == 4);
  BOOST_CHECK(p.channel<B>() == 2);

  // Subtraction.
  p -= p1;
  BOOST_CHECK(p == p1);
}

BOOST_AUTO_TEST_SUITE_END()
