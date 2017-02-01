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

#include <DO/Sara/ImageProcessing/DataAugmentation.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestDataAugmentation, test_zoom)
{
  auto t = ImageDataTransform{};
  t.set_zoom(1.2f);
}


TEST(TestDataAugmentation, test_shift)
{
  auto t = ImageDataTransform{};
  t.set_shift(Vector2i::Ones());
}

TEST(TestDataAugmentation, test_flip)
{
  auto t = ImageDataTransform{};
  t.set_flip(ImageDataTransform::Horizontal);
}

TEST(TestDataAugmentation, test_fancy_pca)
{
  auto t = ImageDataTransform{};
  t.set_fancy_pca(Vector3d::Ones());
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
