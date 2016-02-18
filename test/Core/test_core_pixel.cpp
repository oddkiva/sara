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

#include <gtest/gtest.h>

#include <DO/Sara/Core/Pixel/ColorSpace.hpp>
#include <DO/Sara/Core/Pixel/Pixel.hpp>


using namespace std;
using namespace DO::Sara;


TEST(Test_Pixel, test_rgb_32f)
{
  typedef Pixel<float, Rgb> Rgb32f;

  Rgb32f red(1., 0, 0);
  EXPECT_EQ(red.channel<R>(), 1.f);
  EXPECT_EQ(red.channel<G>(), 0.f);
  EXPECT_EQ(red.channel<B>(), 0.f);
  EXPECT_EQ(red.num_channels(), 3);
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
