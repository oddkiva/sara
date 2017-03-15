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

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/ImageProcessing/ColorFancyPCA.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestColorFancyPca, test)
{
  auto image = Image<Rgb32f>{2, 2};
  image.matrix().fill(Rgb32f::Zero());

  auto alpha = Vector3f::Ones().eval();

  auto fancy_pca = ColorFancyPCA{Matrix3f::Identity(), Vector3f::Ones()};
  fancy_pca(image, alpha);

  for (int i = 0; i < 3; ++i)
    ASSERT_MATRIX_EQ(to_cwh_tensor(image)[0].matrix(), Matrix2f::Ones());
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
