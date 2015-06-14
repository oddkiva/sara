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

#define _USE_MATH_DEFINES

#include <gtest/gtest.h>

#include <DO/Sara/ImageProcessing/SecondMomentMatrix.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestSecondMomentMatrix, test_second_moment_matrix)
{
  Image<Vector2f> vector_field(3, 3);
  vector_field.matrix().fill(Vector2f::Ones());

  Image<Matrix2f> true_moments(3, 3);
  true_moments.array().fill(Matrix2f::Ones());

  Image<Matrix2f> moments;
  moments = vector_field.compute<SecondMomentMatrix>();
  for (size_t i = 0; i != moments.size(); ++i)
    EXPECT_MATRIX_NEAR(true_moments.array()[i], moments.array()[i], 1e-3);
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}