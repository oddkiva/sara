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

#include <DO/Sara/Core/ArrayIterators.hpp>

#include "../AssertHelpers.hpp"


using namespace DO::Sara;
using namespace std;


TEST(TestStrideComputer, test_row_major_strides_computation_2d)
{
  auto sizes = Vector2i{ 10, 20 };
  auto strides = Vector2i{ 20, 1 };

  EXPECT_EQ(StrideComputer<RowMajor>::eval(sizes), strides);
}

TEST(TestStrideComputer, test_col_major_strides_computation_2d)
{
  auto sizes = Vector2i{ 10, 20 };
  auto strides = Vector2i{ 1, 10 };

  EXPECT_EQ(StrideComputer<ColMajor>::eval(sizes), strides);
}

TEST(TestStrideComputer, test_row_major_stride_computation_3d)
{
  auto sizes = Vector3i{ 10, 20, 30 };
  auto strides = Vector3i{ 20*30, 30, 1 };

  EXPECT_EQ(StrideComputer<RowMajor>::eval(sizes), strides);
}

TEST(TestStrideComputer, test_col_major_stride_computation_3d)
{
  auto sizes = Vector3i{ 10, 20, 30 };
  auto strides = Vector3i{ 1, 10, 10*20 };

  EXPECT_EQ(StrideComputer<ColMajor>::eval(sizes), strides);
}


TEST(TestJump, test_jump_2d)
{
  auto coords = Vector2i{ 2, 3 };
  auto sizes = Vector2i{ 10, 20 };
  auto strides = StrideComputer<RowMajor>::eval(sizes);

  EXPECT_EQ(2*20+3, jump(coords, strides));
}

TEST(TestJump, test_jump_3d)
{
  auto coords = Vector3i{ 2, 3, 4 };
  auto sizes = Vector3i{ 10, 20, 30 };
  auto strides = StrideComputer<RowMajor>::eval(sizes);

  EXPECT_EQ(jump(coords, strides), 2*20*30+3*30+4);
}


TEST(TestPositionIncrementer, test_row_major_incrementer_2d)
{
  auto stop = false;
  auto start = Vector2i{ 2, 3 };
  auto end = Vector2i{ 5, 10 };

  auto coords = start;
  for (auto i = start(0); i < end(0); ++i)
  {
    for (auto j = start(1); j < end(1); ++j)
    {
      ASSERT_FALSE(stop);
      ASSERT_MATRIX_EQ(coords, Vector2i(i,j));
      PositionIncrementer<RowMajor>::apply(coords, stop, start, end);
    }
  }
  EXPECT_TRUE(stop);
}

TEST(TestPositionIncrementer, test_col_major_incrementer_2d)
{
  auto stop = false;
  auto start = Vector2i{ 2, 3 };
  auto end = Vector2i{ 5, 10 };

  auto coords = start;
  for (auto j = start(1); j < end(1); ++j)
  {
    for (auto i = start(0); i < end(0); ++i)
    {
      ASSERT_FALSE(stop);
      ASSERT_MATRIX_EQ(coords, Vector2i(i,j));
      PositionIncrementer<ColMajor>::apply(coords, stop, start, end);
    }
  }
  EXPECT_TRUE(stop);
}

TEST(TestPositionDecrementer, test_row_major_decrementer_2d)
{
  auto stop = false;
  auto start = Vector2i{ 2, 3 };
  auto end = Vector2i{ 5, 10 };

  auto coords = Vector2i{};
  coords.array() = end.array()-1;
  for (auto i = end(0)-1; i >= start(0); --i)
  {
    for (auto j = end(1)-1; j >= start(1); --j)
    {
      ASSERT_FALSE(stop);
      ASSERT_MATRIX_EQ(coords, Vector2i(i,j));
      PositionDecrementer<RowMajor>::apply(coords, stop, start, end);
    }
  }
  EXPECT_TRUE(stop);
}

TEST(TestPositionDecrementer, test_col_major_decrementer_2d)
{
  bool stop = false;
  auto start = Vector2i{ 2, 3 };
  auto end = Vector2i{ 5, 10 };

  auto coords = Vector2i{};
  coords.array() = end.array()-1;
  for (int j = end(1)-1; j >= start(1); --j) {
    for (int i = end(0)-1; i >= start(0); --i) {
      ASSERT_FALSE(stop);
      ASSERT_MATRIX_EQ(coords, Vector2i(i,j));
      PositionDecrementer<ColMajor>::apply(coords, stop, start, end);
    }
  }
  EXPECT_TRUE(stop);
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
