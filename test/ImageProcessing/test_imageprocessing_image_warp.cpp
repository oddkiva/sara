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

#include <exception>

#include <gtest/gtest.h>

#include <DO/Sara/Core/Pixel/Typedefs.hpp>
#include <DO/Sara/ImageProcessing/Warp.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


template <class ChannelType>
class TestImageWarp : public testing::Test {};

typedef testing::Types<float, double> ChannelTypes;

TYPED_TEST_CASE_P(TestImageWarp);

TYPED_TEST_P(TestImageWarp, test_image_warp)
{
  typedef TypeParam T;
  Image<T> src(3, 3);
  src.matrix() <<
    0, 1, 2,
    3, 4, 5,
    6, 7, 8;

  Matrix3d homography;
  homography <<
    1, 0, 0,
    0, 1, 0,
    0, 0, 1;

  Image<T> dst(3, 3);
  warp(src, dst, homography);
  EXPECT_MATRIX_NEAR(src.matrix(), dst.matrix(), 1e-7);
}

REGISTER_TYPED_TEST_CASE_P(TestImageWarp, test_image_warp);
INSTANTIATE_TYPED_TEST_CASE_P(DO_Sara_ImageProcessing_Warp,
                              TestImageWarp, ChannelTypes);


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}