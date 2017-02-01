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

#include <vector>

#include <gtest/gtest.h>

#include <DO/Sara/Core/Tensor.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestConversionImageToTensor, test_grayscale_case)
{
  auto image = Image<float>{2, 3};
  image.matrix() <<
    0, 1,
    2, 3,
    4, 5;

  const auto tensor = to_cwh_tensor(image);

  auto true_tensor = Tensor<float, 2>{3, 2};
  true_tensor.matrix() <<
    0, 1,
    2, 3,
    4, 5;

  EXPECT_MATRIX_EQ(true_tensor.matrix(), tensor.matrix());
}

TEST(TestConversionImageToTensor, test_color_case)
{
  auto image = Image<Rgb32f>{2, 3};
  auto m = image.matrix();
  m.fill(Rgb32f{1.,2.,3.});

  m(0,0) *= 0; m(0,1) *= 1;
  m(1,0) *= 2; m(1,1) *= 3;
  m(2,0) *= 4; m(2,1) *= 5;

  const auto tensor = to_cwh_tensor(image);
  const auto r = tensor[0].matrix();
  const auto g = tensor[1].matrix();
  const auto b = tensor[2].matrix();

  auto true_r = Tensor<float, 2>{3, 2};
  auto true_g = Tensor<float, 2>{3, 2};
  auto true_b = Tensor<float, 2>{3, 2};
  true_r.matrix() <<
    0, 1,
    2, 3,
    4, 5;
  true_g.matrix() <<
    0,  2,
    4,  6,
    8, 10;
  true_b.matrix() <<
     0,  3,
     6,  9,
    12, 15;

  EXPECT_MATRIX_EQ(true_r.matrix(), r);
  EXPECT_MATRIX_EQ(true_g.matrix(), g);
  EXPECT_MATRIX_EQ(true_b.matrix(), b);
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
