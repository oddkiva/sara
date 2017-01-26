// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <gtest/gtest.h>

#include <DO/Sara/Core/Image/Operations.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


class TestImageCrop : public testing::Test
{
protected:
  Image<float> image;
  Vector2i sizes;

  TestImageCrop()
  {
    sizes << 3, 4;
    image.resize(sizes);
    image.matrix() <<
      1, 1, 1,
      2, 2, 2,
      3, 3, 3,
      4, 4, 4;
  }
};


TEST_F(TestImageCrop, test_safe_crop_image_within_bounds)
{
  auto a = Vector2i{ 1, 1 };
  auto b = Vector2i{ 3, 2 };
  auto safe_cropped_image = safe_crop(image, a, b);
  auto cropped_image = crop(image, a, b);

  auto true_cropped_image = Image<float, 2>{ 2, 1 };
  true_cropped_image.matrix() << 2, 2;
  EXPECT_MATRIX_EQ(true_cropped_image.sizes(), safe_cropped_image.sizes());
  EXPECT_MATRIX_EQ(true_cropped_image.matrix(), safe_cropped_image.matrix());
  EXPECT_MATRIX_EQ(true_cropped_image.sizes(), cropped_image.sizes());
  EXPECT_MATRIX_EQ(true_cropped_image.matrix(), cropped_image.matrix());

  auto x = 1, y = 1;
  auto w = 2, h = 1;
  safe_cropped_image = safe_crop(image, x, y, w, h);
  cropped_image = crop(image, x, y, w, h);
  EXPECT_MATRIX_EQ(true_cropped_image.sizes(), safe_cropped_image.sizes());
  EXPECT_MATRIX_EQ(true_cropped_image.matrix(), safe_cropped_image.matrix());
  EXPECT_MATRIX_EQ(true_cropped_image.sizes(), cropped_image.sizes());
  EXPECT_MATRIX_EQ(true_cropped_image.matrix(), cropped_image.matrix());
}


TEST_F(TestImageCrop, test_safe_crop_image_out_of_bounds_1)
{
  auto a = Vector2i{ -3, -3 };
  auto b = Vector2i{ 0, 0 };
  auto cropped_image = safe_crop(image, a, b);

  auto true_cropped_image = Image<float, 2>{ 3, 3 };
  true_cropped_image.matrix().fill(0);
  EXPECT_MATRIX_EQ(true_cropped_image.sizes(), cropped_image.sizes());
  EXPECT_MATRIX_EQ(true_cropped_image.matrix(), cropped_image.matrix());

  auto x = -3, y = -3;
  auto w =  3, h =  3;
  cropped_image = safe_crop(image, x, y, w, h);
  EXPECT_MATRIX_EQ(true_cropped_image.sizes(), cropped_image.sizes());
  EXPECT_MATRIX_EQ(true_cropped_image.matrix(), cropped_image.matrix());

  auto cx = -2, cy = -2, r = 1;
  cropped_image = safe_crop(image, cx, cy, r);
  EXPECT_MATRIX_EQ(true_cropped_image.sizes(), cropped_image.sizes());
  EXPECT_MATRIX_EQ(true_cropped_image.matrix(), cropped_image.matrix());
}


TEST_F(TestImageCrop, test_safe_crop_image_out_of_bounds_2)
{
  auto a = Vector2i{ -1, -1 };
  auto b = Vector2i{ 2, 3 };
  auto cropped_image = safe_crop(image, a, b);

  auto true_cropped_image = Image<float, 2>{ 3, 4 };
  true_cropped_image.matrix() <<
    0, 0, 0,
    0, 1, 1,
    0, 2, 2,
    0, 3, 3;
  EXPECT_MATRIX_EQ(true_cropped_image.sizes(), cropped_image.sizes());
  EXPECT_MATRIX_EQ(true_cropped_image.matrix(), cropped_image.matrix());

  auto x = -1, y = -1;
  auto w =  3, h =  4;
  cropped_image = safe_crop(image, x, y, w, h);
  EXPECT_MATRIX_EQ(true_cropped_image.sizes(), cropped_image.sizes());
  EXPECT_MATRIX_EQ(true_cropped_image.matrix(), cropped_image.matrix());
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
