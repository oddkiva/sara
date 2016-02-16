// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <stdint.h>

#include <gtest/gtest.h>

#include "Core/Pixel/PixelTraits.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestPixelTraits, test_pixel_traits_min_max_value)
{
  EXPECT_EQ(numeric_limits<int>::min(), PixelTraits<int>::min());
  EXPECT_EQ(0, PixelTraits<float>::min());

  EXPECT_EQ(std::numeric_limits<int>::max(), PixelTraits<int>::max());
  EXPECT_EQ(1, PixelTraits<float>::max());

  Matrix<uint8_t, 3, 1> expected_black = Matrix<uint8_t, 3, 1>::Zero();
  Matrix<uint8_t, 3, 1> actual_black =
    PixelTraits<Matrix<uint8_t, 3, 1> >::min();
  EXPECT_EQ(expected_black, actual_black);

  Vector3f expected_zeros = Vector3f::Zero();
  Vector3f actual_zeros = PixelTraits<Vector3f>::min();
  EXPECT_EQ(expected_zeros, actual_zeros);

  Matrix<uint8_t, 3, 1> expected_black_3 = Matrix<uint8_t, 3, 1>::Zero();
  Matrix<uint8_t, 3, 1> actual_black_3 =
    PixelTraits<Matrix<uint8_t, 3, 1> >::min();
  EXPECT_EQ(expected_black_3, actual_black_3);

  Vector3f expected_zeros_3 = Vector3f::Zero();
  Vector3f actual_zeros_3 = PixelTraits<Vector3f>::min();
  EXPECT_EQ(expected_zeros_3, actual_zeros_3);

  Vector3f expected_ones_3 = Vector3f::Ones();
  Vector3f actual_ones_3 = PixelTraits<Vector3f>::max();
  EXPECT_EQ(expected_ones_3, actual_ones_3);
}


int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
