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

#include <DO/Sara/ImageProcessing/Flip.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


class TestImageFlip : public testing::Test
{
protected:
  Image<int> image;


  TestImageFlip() : testing::Test()
  {
    // Draw an 'F' letter.
    image.resize(4, 6);
    image.matrix() <<
      1, 1, 1, 1,
      1, 0, 0, 0,
      1, 1, 1, 0,
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 0, 0, 0;
  }

  virtual ~TestImageFlip() {}
};


TEST_F(TestImageFlip, test_flip_horizontally)
{
  auto true_flipped_image = Image<int>{4, 6};
  true_flipped_image.matrix() <<
      1, 1, 1, 1,
      0, 0, 0, 1,
      0, 1, 1, 1,
      0, 0, 0, 1,
      0, 0, 0, 1,
      0, 0, 0, 1;

  flip_horizontally(image);
  ASSERT_MATRIX_EQ(true_flipped_image.matrix(), image.matrix());
}

TEST_F(TestImageFlip, test_flip_vertically)
{
  auto true_flipped_image = Image<int>{4, 6};
  true_flipped_image.matrix() <<
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 1, 1, 0,
      1, 0, 0, 0,
      1, 1, 1, 1;

  flip_vertically(image);
  ASSERT_MATRIX_EQ(true_flipped_image.matrix(), image.matrix());
}

TEST_F(TestImageFlip, test_transpose)
{
  auto true_flipped_image = Image<int>{6, 4};
  true_flipped_image.matrix() <<
      1, 1, 1, 1, 1, 1,
      1, 0, 1, 0, 0, 0,
      1, 0, 1, 0, 0, 0,
      1, 0, 0, 0, 0, 0;

  transpose(image);
  ASSERT_MATRIX_EQ(true_flipped_image.matrix(), image.matrix());
}

TEST_F(TestImageFlip, test_transverse)
{
  auto true_flipped_image = Image<int>{6, 4};
  true_flipped_image.matrix() <<
      0, 0, 0, 0, 0, 1, 
      0, 0, 0, 1, 0, 1, 
      0, 0, 0, 1, 0, 1, 
      1, 1, 1, 1, 1, 1; 

  transverse(image);
  ASSERT_MATRIX_EQ(true_flipped_image.matrix(), image.matrix());
}

TEST_F(TestImageFlip, test_rotate_ccw_90)
{
  auto true_flipped_image = Image<int>{6, 4};
  true_flipped_image.matrix() <<
    1, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 0, 0,
    1, 1, 1, 1, 1, 1;

  rotate_ccw_90(image);
  ASSERT_MATRIX_EQ(true_flipped_image.matrix(), image.matrix());
}

TEST_F(TestImageFlip, test_rotate_ccw_180)
{
  auto true_flipped_image = Image<int>{4, 6};
  true_flipped_image.matrix() <<
      0, 0, 0, 1,
      0, 0, 0, 1,
      0, 0, 0, 1,
      0, 1, 1, 1,
      0, 0, 0, 1,
      1, 1, 1, 1;

  rotate_ccw_180(image);
  ASSERT_MATRIX_EQ(true_flipped_image.matrix(), image.matrix());
}

TEST_F(TestImageFlip, test_rotate_ccw_270)
{
  auto true_flipped_image = Image<int>{6, 4};
  true_flipped_image.matrix() <<
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 1, 0, 1,
    0, 0, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 1;

  rotate_ccw_270(image);
  ASSERT_MATRIX_EQ(true_flipped_image.matrix(), image.matrix());
}

TEST_F(TestImageFlip, test_rotate_cw_270)
{
  auto true_flipped_image = Image<int>{6, 4};
  true_flipped_image.matrix() <<
    1, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 0, 0,
    1, 1, 1, 1, 1, 1;

  rotate_cw_270(image);
  ASSERT_MATRIX_EQ(true_flipped_image.matrix(), image.matrix());
}

TEST_F(TestImageFlip, test_rotate_cw_180)
{
  auto true_flipped_image = Image<int>{4, 6};
  true_flipped_image.matrix() <<
      0, 0, 0, 1,
      0, 0, 0, 1,
      0, 0, 0, 1,
      0, 1, 1, 1,
      0, 0, 0, 1,
      1, 1, 1, 1;

  rotate_cw_180(image);
  ASSERT_MATRIX_EQ(true_flipped_image.matrix(), image.matrix());
}

TEST_F(TestImageFlip, test_rotate_cw_90)
{
  auto true_flipped_image = Image<int>{6, 4};
  true_flipped_image.matrix() <<
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 1, 0, 1,
    0, 0, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 1;

  rotate_cw_90(image);
  ASSERT_MATRIX_EQ(true_flipped_image.matrix(), image.matrix());
}
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
