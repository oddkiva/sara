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

#include <DO/Sara/ImageProcessing/Orientation.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestOrientation, test_orientation)
{
  Image<Vector2f> vector_field(3, 3);
  vector_field.matrix().fill(Vector2f::Ones());

  Image<float> true_orientations(3, 3);
  true_orientations.array().fill(static_cast<float>(M_PI_4));

  Image<float> orientations;

  orientation(vector_field, orientations);
  EXPECT_MATRIX_NEAR(true_orientations.matrix(), orientations.matrix(), 1e-3);

  orientations = orientation(vector_field);
  EXPECT_MATRIX_NEAR(true_orientations.matrix(), orientations.matrix(), 1e-3);

  orientations = vector_field.compute<Orientation>();
  EXPECT_MATRIX_NEAR(true_orientations.matrix(), orientations.matrix(), 1e-3);
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}