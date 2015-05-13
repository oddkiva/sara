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

#include <gtest/gtest.h>

#include <DO/Sara/Core/Image/Image.hpp>


using namespace std;
using namespace DO;


TEST(TestImageClass, test_2d_image_constructor)
{
  Image<int> image(10, 20);
  EXPECT_EQ(image.width(), 10);
  EXPECT_EQ(image.height(), 20);

  Image<int, 3> volume(5, 10, 20);
  EXPECT_EQ(volume.width(), 5);
  EXPECT_EQ(volume.height(), 10);
  EXPECT_EQ(volume.depth(), 20);

  Image<int, 3> volume2;
  volume2 = volume;
  EXPECT_EQ(volume2.width(), 5);
  EXPECT_EQ(volume2.height(), 10);
  EXPECT_EQ(volume2.depth(), 20);
}


TEST(TestImageClass, test_matrix_view)
{
  Image<int> A(2, 3);
  A.matrix() <<
    1, 2,
    3, 4,
    5, 6;

  EXPECT_EQ(A(0, 0), 1);
  EXPECT_EQ(A(1, 0), 2);
  EXPECT_EQ(A(0, 1), 3);
  EXPECT_EQ(A(1, 1), 4);
  EXPECT_EQ(A(0, 2), 5);
  EXPECT_EQ(A(1, 2), 6);
}


// ========================================================================== //
// Run the tests.
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
