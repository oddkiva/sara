// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <limits>
#include <stdint.h>
#include <vector>

#include <gtest/gtest.h>

#include <DO/Core/Pixel/PackedPixel.hpp>


using namespace std;
using namespace DO;


TEST(Test_PackedPixelBase_3, test_initialization)
{
  typedef PackedPixelBase_3<uint16_t, 5, 6, 5> pixel_base_type;
  ASSERT_EQ(sizeof(pixel_base_type), 2u);

  pixel_base_type red = {31, 0, 0};
  ASSERT_EQ(*reinterpret_cast<uint16_t *>(&red), 0x001f);
  ASSERT_EQ(red.channel_0, 31);
  ASSERT_EQ(red.channel_1, 0);
  ASSERT_EQ(red.channel_2, 0);

  pixel_base_type green = {0, 63, 0};
  ASSERT_EQ(*reinterpret_cast<uint16_t *>(&green), 0x07e0);
  ASSERT_EQ(green.channel_0, 0);
  ASSERT_EQ(green.channel_1, 63);
  ASSERT_EQ(green.channel_2, 0);

  pixel_base_type blue = {0, 0, 31};
  ASSERT_EQ(*reinterpret_cast<uint16_t *>(&blue), 0xf800);
  ASSERT_EQ(blue.channel_0, 0);
  ASSERT_EQ(blue.channel_1, 0);
  ASSERT_EQ(blue.channel_2, 31);

  pixel_base_type white = {31, 63, 31};
  ASSERT_EQ(*reinterpret_cast<uint16_t *>(&white), 0xffff);
  ASSERT_EQ(white.channel_0, 31);
  ASSERT_EQ(white.channel_1, 63);
  ASSERT_EQ(white.channel_2, 31);

  pixel_base_type black = {0, 0, 0};
  ASSERT_EQ(*reinterpret_cast<uint16_t *>(&black), 0x0);
  ASSERT_EQ(black.channel_0, 0);
  ASSERT_EQ(black.channel_1, 0);
  ASSERT_EQ(black.channel_2, 0);
}


TEST(Test_PackedPixelBase_4, test_initialization)
{
  typedef PackedPixelBase_4<uint8_t, 8, 8, 8, 8> pixel_base_type;
  ASSERT_EQ(sizeof(pixel_base_type), 4);

  pixel_base_type cyan = {255, 0, 0, 0};
  ASSERT_EQ(*reinterpret_cast<uint32_t *>(&cyan), 0xff);
  ASSERT_EQ(cyan.channel_0, 255);
  ASSERT_EQ(cyan.channel_1, 0);
  ASSERT_EQ(cyan.channel_2, 0);

  pixel_base_type magenta = {0, 255, 0, 0};
  ASSERT_EQ(*reinterpret_cast<uint16_t *>(&magenta), 0xff00);
  ASSERT_EQ(magenta.channel_0, 0);
  ASSERT_EQ(magenta.channel_1, 255);
  ASSERT_EQ(magenta.channel_2, 0);

  pixel_base_type yellow = {0, 0, 255, 0};
  ASSERT_EQ(*reinterpret_cast<uint32_t *>(&yellow), 0x00ff0000);
  ASSERT_EQ(yellow.channel_0, 0);
  ASSERT_EQ(yellow.channel_1, 0);
  ASSERT_EQ(yellow.channel_2, 255);

  pixel_base_type black = {0, 0, 0, 255};
  ASSERT_EQ(*reinterpret_cast<uint32_t *>(&black), 0xff000000);
  ASSERT_EQ(black.channel_0, 0);
  ASSERT_EQ(black.channel_1, 0);
  ASSERT_EQ(black.channel_2, 0);
  ASSERT_EQ(black.channel_3, 255);

  pixel_base_type white = {255, 255, 255, 255};
  ASSERT_EQ(*reinterpret_cast<uint32_t *>(&white), 0xffffffff);
  ASSERT_EQ(white.channel_0, 255);
  ASSERT_EQ(white.channel_1, 255);
  ASSERT_EQ(white.channel_2, 255);
  ASSERT_EQ(white.channel_3, 255);
}


TEST(Test_ChannelIndex, test_channel_index)
{
  typedef ColorModel<Rgb, Meta::IntArray_3<2,1,0> > BgrModel;
  int r = ChannelIndex<BgrModel, R>::value;
  int g = ChannelIndex<BgrModel, G>::value;
  int b = ChannelIndex<BgrModel, B>::value;
  ASSERT_EQ(r, 2);
  ASSERT_EQ(g, 1);
  ASSERT_EQ(b, 0);
}


TEST(Test_PackedPixel_3, test_packed_pixel)
{
  typedef PackedPixelBase_3<uint16_t, 5, 6, 5> Pixel_565;
  typedef ColorModel<Rgb, Meta::IntArray_3<2,1,0> > BgrModel;
  typedef PackedPixel<Pixel_565, BgrModel> Bgr_565;

  // Constructor.
  Bgr_565 p1(1, 2, 0);
  ASSERT_EQ(p1.channel<B>(), 1);
  ASSERT_EQ(p1.channel<G>(), 2);
  ASSERT_EQ(p1.channel<R>(), 0);

  // Constructor.
  Bgr_565 p2(1, 3, 1);
  ASSERT_EQ(p2.channel<B>(), 1);
  ASSERT_EQ(p2.channel<G>(), 3);
  ASSERT_EQ(p2.channel<R>(), 1);

  // Assignment.
  p2 = p1;
  ASSERT_EQ(p2.channel<B>(), 1);
  ASSERT_EQ(p2.channel<G>(), 2);
  ASSERT_EQ(p2.channel<R>(), 0);

  // Copy constructor.
  Bgr_565 p(p1);
  ASSERT_EQ(p.channel<B>(), 1);
  ASSERT_EQ(p.channel<G>(), 2);
  ASSERT_EQ(p.channel<R>(), 0);

  // Addition.
  p = p1 + p2;
  ASSERT_EQ(p.channel<R>(), 0);
  ASSERT_EQ(p.channel<G>(), 4);
  ASSERT_EQ(p.channel<B>(), 2);

  // Subtraction.
  p -= p1;
  ASSERT_EQ(p, p1);
}


int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}