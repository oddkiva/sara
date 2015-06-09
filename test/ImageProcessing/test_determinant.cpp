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

#include <DO/Sara/ImageProcessing/Determinant.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestDeterminant, test_determinant)
{
  Image<Matrix2f> tensor(3, 3);
  tensor.array().fill(Matrix2f::Ones());

  Image<float> det;
  det = tensor.compute<Determinant>();

  for (int i = 0; i < det.array().size(); ++i)
    EXPECT_NEAR(0, det.array()[i], 1e-5);
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}