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
#include <DO/Sara/ImageProcessing/DataAugmentation.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestDataAugmentation, test_zoom)
{
  auto t = ImageDataTransform{};
  t.set_zoom(2.f);
  t.out_sizes = Vector2i::Ones() * 2;

  auto in = Image<float>{3, 3};
  in.matrix() <<
    0, 1, 2,
    3, 4, 5,
    6, 7, 8;

  auto out = t.extract_patch(in);
  ASSERT_MATRIX_EQ(out.sizes(), t.out_sizes);
}


TEST(TestDataAugmentation, test_shift)
{
  auto t = ImageDataTransform{};
  t.set_shift(Vector2i::Ones());
  t.out_sizes = Vector2i::Ones() * 2;

  auto in = Image<int>{3, 3};
  in.matrix() <<
    0, 1, 2,
    3, 4, 5,
    6, 7, 8;

  auto out = t.extract_patch(in);

  auto true_out = Image<int>{2, 2};
  true_out.matrix() <<
    4, 5,
    7, 8;

  ASSERT_MATRIX_EQ(true_out.matrix(), out.matrix());
}

TEST(TestDataAugmentation, test_flip)
{
  auto t = ImageDataTransform{};
  t.set_flip(ImageDataTransform::Horizontal);
}

TEST(TestDataAugmentation, test_fancy_pca)
{
  auto t = ImageDataTransform{};
  t.set_fancy_pca(Vector3f::Ones());
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
