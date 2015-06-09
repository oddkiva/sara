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

#include <DO/Sara/ImageProcessing/Norm.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


class TestNorms : public testing::Test
{
protected:
  Image<Vector2f> vector_field;

  TestNorms() : testing::Test()
  {
    vector_field.resize(3, 3);
    vector_field.matrix().fill(Vector2f::Ones());
  }
};


TEST_F(TestNorms, test_squared_norm)
{
  Image<float> true_squared_norm_image(3, 3);
  true_squared_norm_image.array().fill(2);

  Image<float> squared_norm_image;

  squared_norm(vector_field, squared_norm_image);
  EXPECT_MATRIX_EQ(true_squared_norm_image.matrix(),
                   squared_norm_image.matrix());

  squared_norm_image = squared_norm(vector_field);
  EXPECT_MATRIX_EQ(true_squared_norm_image.matrix(),
                   squared_norm_image.matrix());

  squared_norm_image = vector_field.compute<SquaredNorm>();
  EXPECT_MATRIX_EQ(true_squared_norm_image.matrix(),
                   squared_norm_image.matrix());
}


TEST_F(TestNorms, test_blue_norm)
{
  Image<float> true_blue_norm_image(3, 3);
  true_blue_norm_image.array().fill(Vector2f::Ones().blueNorm());

  Image<float> blue_norm_image;

  blue_norm(vector_field, blue_norm_image);
  EXPECT_MATRIX_EQ(true_blue_norm_image.matrix(),
                   blue_norm_image.matrix());

  blue_norm_image = blue_norm(vector_field);
  EXPECT_MATRIX_EQ(true_blue_norm_image.matrix(),
                   blue_norm_image.matrix());

  blue_norm_image = vector_field.compute<BlueNorm>();
  EXPECT_MATRIX_EQ(true_blue_norm_image.matrix(),
                   blue_norm_image.matrix());
}


TEST_F(TestNorms, test_stable_norm)
{
  Image<float> true_stable_norm_image(3, 3);
  true_stable_norm_image.array().fill(Vector2f::Ones().stableNorm());

  Image<float> stable_norm_image;

  stable_norm(vector_field, stable_norm_image);
  EXPECT_MATRIX_EQ(true_stable_norm_image.matrix(),
                   stable_norm_image.matrix());

  stable_norm_image = stable_norm(vector_field);
  EXPECT_MATRIX_EQ(true_stable_norm_image.matrix(),
                   stable_norm_image.matrix());

  stable_norm_image = vector_field.compute<StableNorm>();
  EXPECT_MATRIX_EQ(true_stable_norm_image.matrix(),
                   stable_norm_image.matrix());
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}