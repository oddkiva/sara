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

#include <gtest/gtest.h>
#include <DO/Core/Locator.hpp>
#include <vld.h>
#include "AssertHelpers.hpp"

using namespace DO;
using namespace std;


TEST(TestStrideComputer, test_row_major_strides_computation_2d)
{
  Vector2i sizes(10, 20);
  Vector2i strides(20, 1);

  EXPECT_EQ(StrideComputer<RowMajor>::eval(sizes), strides);
}

TEST(TestStrideComputer, test_col_major_strides_computation_2d)
{
  Vector2i sizes(10, 20);
  Vector2i strides(1, 10);

  EXPECT_EQ(StrideComputer<ColMajor>::eval(sizes), strides);
}

TEST(TestStrideComputer, test_row_major_stride_computation_3d)
{
  Vector3i sizes(10, 20, 30);
  Vector3i strides(20*30, 30, 1);

  EXPECT_EQ(StrideComputer<RowMajor>::eval(sizes), strides);
}

TEST(TestStrideComputer, test_col_major_stride_computation_3d)
{
  Vector3i sizes(10, 20, 30);
  Vector3i strides(1, 10, 10*20);

  EXPECT_EQ(StrideComputer<ColMajor>::eval(sizes), strides);
}


TEST(TestJump, test_jump_2d)
{
  Vector2i coords(2, 3);
  Vector2i sizes(10, 20);
  Vector2i strides = StrideComputer<RowMajor>::eval(sizes);

  EXPECT_EQ(2*20+3, jump(coords, strides));
}

TEST(TestJump, test_jump_3d)
{
  Vector3i coords(2, 3, 4);
  Vector3i sizes(10, 20, 30);
  Vector3i strides = StrideComputer<RowMajor>::eval(sizes);

  EXPECT_EQ(jump(coords, strides), 2*20*30+3*30+4);
}


TEST(TestPositionIncrementer, test_row_major_incrementer_2d)
{
  bool stop = false;
  Vector2i start(2, 3);
  Vector2i end(5, 10);
  
  Vector2i coords(start);
  for (int i = start(0); i < end(0); ++i) {
    for (int j = start(1); j < end(1); ++j) {
      ASSERT_FALSE(stop);
      ASSERT_MATRIX_EQ(coords, Vector2i(i,j));
      PositionIncrementer<RowMajor>::apply(coords, stop, start, end);
    }
  }
  cout << coords << endl;
  EXPECT_TRUE(stop);
}

TEST(TestPositionIncrementer, test_col_major_incrementer_2d)
{
  bool stop = false;
  Vector2i start(2, 3);
  Vector2i end(5, 10);

  Vector2i coords(start);
  for (int j = start(1); j < end(1); ++j) {
    for (int i = start(0); i < end(0); ++i) {
      ASSERT_FALSE(stop);
      ASSERT_MATRIX_EQ(coords, Vector2i(i,j));
      PositionIncrementer<ColMajor>::apply(coords, stop, start, end);
    }
  }
  cout << coords << endl;
  EXPECT_TRUE(stop);
}

TEST(TestPositionDecrementer, test_row_major_decrementer_2d)
{
  bool stop = false;
  Vector2i start(2, 3);
  Vector2i end(5, 10);

  Vector2i coords;
  coords.array() = end.array()-1;
  for (int i = end(0)-1; i >= start(0); --i) {
    for (int j = end(1)-1; j >= start(1); --j) {
      ASSERT_FALSE(stop);
      ASSERT_MATRIX_EQ(coords, Vector2i(i,j));
      PositionDecrementer<RowMajor>::apply(coords, stop, start, end);
    }
  }
  cout << coords << endl;
  EXPECT_TRUE(stop);
}

TEST(TestPositionDecrementer, test_col_major_decrementer_2d)
{
  bool stop = false;
  Vector2i start(2, 3);
  Vector2i end(5, 10);

  Vector2i coords;
  coords.array() = end.array()-1;
  for (int j = end(1)-1; j >= start(1); --j) {
    for (int i = end(0)-1; i >= start(0); --i) {
      ASSERT_FALSE(stop);
      ASSERT_MATRIX_EQ(coords, Vector2i(i,j));
      PositionDecrementer<ColMajor>::apply(coords, stop, start, end);
    }
  }
  cout << coords << endl;
  EXPECT_TRUE(stop);
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}