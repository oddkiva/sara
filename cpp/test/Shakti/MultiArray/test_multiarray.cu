// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <gtest/gtest.h>

#include <DO/Shakti/MultiArray.hpp>


namespace shakti = DO::Shakti;

using namespace std;


TEST(TestMultiArray, test_constructor_1d)
{
  shakti::Array<float> array{ 10 };
  EXPECT_EQ(10, array.sizes());
}

TEST(TestMultiArray, test_constructor_2d)
{
  shakti::MultiArray<float, 2> matrix{ { 3, 4 } };
  EXPECT_EQ(shakti::Vector2i(3, 4), matrix.sizes());
  EXPECT_EQ(3, matrix.size(0));
  EXPECT_EQ(4, matrix.size(1));

  EXPECT_EQ(3, matrix.width());
  EXPECT_EQ(4, matrix.height());
}

TEST(TestMultiArray, test_constructor_3d)
{
  shakti::MultiArray<float, 3> matrix{ { 3, 4, 5 } };
  EXPECT_EQ(shakti::Vector3i(3, 4, 5), matrix.sizes());
  EXPECT_EQ(3, matrix.size(0));
  EXPECT_EQ(4, matrix.size(1));
  EXPECT_EQ(5, matrix.size(2));

  EXPECT_EQ(3, matrix.width());
  EXPECT_EQ(4, matrix.height());
  EXPECT_EQ(5, matrix.depth());
}

TEST(MultiArray, test_copy_between_host_and_device_2d)
{
  const int w = 3;
  const int h = 4;
  float in_host_data[] = {
    0, 1, 2,
    3, 4, 5,
    6, 7, 8,
    9, 10, 11
  };

  // Copy to device.
  shakti::MultiArray<float, 2> out_device_image{ in_host_data, { w, h } };

  // Copy back to host.
  float out_host_data[w*h];
  out_device_image.copy_to_host(out_host_data);

  EXPECT_TRUE(equal(in_host_data, in_host_data + w*h, out_host_data));
}

TEST(MultiArray, test_copy_between_host_and_device_3d)
{
  const int w = 3;
  const int h = 4;
  const int d = 5;
  float in_host_data[w*h*d];
  for (int i = 0; i < w*h*d; ++i)
    in_host_data[i] = i;

  // Copy to device.
  shakti::MultiArray<float, 3> out_device_image{ in_host_data, { w, h, d } };

  // Copy back to host.
  float out_host_data[w*h*d];
  out_device_image.copy_to_host(out_host_data);

  EXPECT_TRUE(equal(in_host_data, in_host_data + w*h*d, out_host_data));
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
