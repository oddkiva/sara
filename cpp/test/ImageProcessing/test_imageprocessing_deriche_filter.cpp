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

#include <DO/Sara/ImageProcessing/Deriche.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestDericheFilter, test_inplace_deriche)
{
  auto signal = Image<float>{10, 10};
  signal.flat_array().setOnes();

  auto true_matrix = MatrixXf(10, 10);
  true_matrix.setOnes();

  auto derivative_order = int{};
  auto derivative_axis = int{};

  // Test the smoothing filter for the two axes.
  derivative_order = 0;
  derivative_axis = 0;
  inplace_deriche(signal, 1.f, derivative_order, derivative_axis);
  EXPECT_MATRIX_NEAR(signal.matrix(), true_matrix, 1e-5);

  derivative_order = 0;
  derivative_axis = 1;
  inplace_deriche(signal, 1.f, derivative_order, derivative_axis);
  EXPECT_MATRIX_NEAR(signal.matrix(), true_matrix, 1e-5);


  // Test the first-order derivative filter for the two axes.
  derivative_order = 1;
  derivative_axis = 0;
  signal.flat_array().fill(1);
  true_matrix.fill(0);
  inplace_deriche(signal, 1.f, derivative_order, derivative_axis);
  EXPECT_MATRIX_NEAR(signal.matrix(), true_matrix, 1e-5);

  derivative_order = 1;
  derivative_axis = 1;
  signal.flat_array().fill(1);
  true_matrix.fill(0);
  inplace_deriche(signal, 1.f, derivative_order, derivative_axis);
  EXPECT_MATRIX_NEAR(signal.matrix(), true_matrix, 1e-5);


  // TODO: test the second-order derivative filter for the two axes.
  // We need to have a good intuition for the second-order derivative though.
  // But let's leave it for another PR.
}


TEST(TestDericheFilter, test_inplace_deriche_blur)
{
  auto signal = Image<float>{10, 10};
  signal.flat_array().setOnes();

  auto true_matrix = MatrixXf(10, 10);
  true_matrix.setOnes();

  const auto sigmas = Vector2f::Ones().eval();
  inplace_deriche_blur(signal, sigmas);
  EXPECT_MATRIX_NEAR(signal.matrix(), true_matrix, 1e-5);

  float sigma = 1;
  inplace_deriche_blur(signal, sigma);
  EXPECT_MATRIX_NEAR(signal.matrix(), true_matrix, 1e-5);
}


TEST(TestDericheFilter, test_convenience_deriche_blur)
{
  auto in_signal = Image<float>{10, 10};
  in_signal.flat_array().setOnes();

  auto true_matrix = MatrixXf(10, 10);
  true_matrix.setOnes();

  auto out_signal = in_signal.compute<DericheBlur>(1, true);
  EXPECT_MATRIX_NEAR(out_signal.matrix(), true_matrix, 1e-5);

  out_signal.clear();
  out_signal = in_signal.compute<DericheBlur>(1);
  EXPECT_MATRIX_NEAR(out_signal.matrix(), true_matrix, 1e-5);
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
