// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //


#include <gtest/gtest.h>

#include <DO/Sara/ImageProcessing/Scaling.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestScaling, test_upscale)
{
  Image<float> src(2, 2);
  src.matrix() <<
    0, 1,
    2, 3;

  Image<float> dst;
  dst = upscale(src, 2);

  Image<float> true_dst(4, 4);
  true_dst.matrix() <<
    0, 0, 1, 1,
    0, 0, 1, 1,
    2, 2, 3, 3,
    2, 2, 3, 3;
  EXPECT_MATRIX_EQ(true_dst.matrix(), dst.matrix());
}


TEST(TestScaling, test_downscale)
{
  Image<float> src(4, 4);
  src.matrix() <<
    0, 0, 1, 1,
    0, 0, 1, 1,
    2, 2, 3, 3,
    2, 2, 3, 3;

  Image<float> dst;
  dst = downscale(src, 2);

  Image<float> true_dst(2, 2);
  true_dst.matrix() <<
    0, 1,
    2, 3;
  EXPECT_MATRIX_EQ(true_dst.matrix(), dst.matrix());
}


TEST(TestScaling, test_enlarge)
{
  Image<float> src(2, 2);
  src.matrix() <<
    0, 1,
    2, 3;

  Image<float> true_dst(4, 4);
  true_dst.matrix() <<
    0, 0.5, 1, 1,
    1, 1.5, 2, 2,
    2, 2.5, 3, 3,
    2, 2.5, 3, 3;

  Image<float> dst;

  dst = enlarge(src, Vector2i(4, 4));
  EXPECT_MATRIX_EQ(true_dst.matrix(), dst.matrix());

  dst = enlarge(src, 4, 4);
  EXPECT_MATRIX_EQ(true_dst.matrix(), dst.matrix());

  dst = enlarge(src, 2);
  EXPECT_MATRIX_EQ(true_dst.matrix(), dst.matrix());
}
 

TEST(TestScaling, test_reduce)
{
  Image<float> src(4, 4);
  src.matrix() <<
    0, 0.5, 1, 1,
    1, 1.5, 2, 2,
    2, 2.5, 3, 3,
    2, 2.5, 3, 3;

  Image<float> true_dst(2, 2);
  true_dst.matrix() <<
    0, 1,
    2, 3;

  Image<float> dst;

  dst = reduce(src, Vector2i(2, 2));
  EXPECT_LE((true_dst.matrix()-dst.matrix()).lpNorm<Infinity>(), 0.4);

  dst = reduce(src, 2, 2);
  EXPECT_LE((true_dst.matrix()-dst.matrix()).lpNorm<Infinity>(), 0.4);

  dst = reduce(src, 2);
  EXPECT_LE((true_dst.matrix()-dst.matrix()).lpNorm<Infinity>(), 0.4);
}

TEST(TestScaling, test_reduce_2)
{
  auto lambda = [](double lambda) {
    return Rgb64f{ lambda, lambda, lambda };
  };
  Image<Rgb64f> src(4, 4);
  src(0, 0) = lambda(0); src(1, 0) = lambda(0.5); src(2, 0) = lambda(1); src(3, 0) = lambda(1);
  src(0, 1) = lambda(1); src(1, 1) = lambda(1.5); src(2, 1) = lambda(2); src(3, 1) = lambda(2);
  src(0, 2) = lambda(2); src(1, 2) = lambda(2.5); src(2, 2) = lambda(3); src(3, 2) = lambda(3);
  src(0, 3) = lambda(2); src(1, 3) = lambda(2.5); src(2, 3) = lambda(3); src(3, 3) = lambda(3);

  Image<Rgb64f> true_dst(2, 2);
  true_dst.matrix();
  true_dst(0, 0) = lambda(0); true_dst(1, 0) = lambda(1);
  true_dst(0, 1) = lambda(2); true_dst(1, 1) = lambda(3);

  Image<Rgb64f> dst;
  dst = reduce(src, 2);

  auto dst_pixel = dst.begin();
  auto true_dst_pixel = true_dst.begin();
  for (; dst_pixel != dst.end(); ++dst_pixel, ++true_dst_pixel)
    EXPECT_LE((*true_dst_pixel - *dst_pixel).lpNorm<Infinity>(), 0.4);
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}