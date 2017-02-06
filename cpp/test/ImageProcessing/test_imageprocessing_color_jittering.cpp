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
#include <DO/Sara/ImageProcessing/ColorJitter.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestNormalDistribution, test_sampling)
{
  auto normal_dist = NormalDistribution{};

  auto samples = vector<float>{};
  auto sample_i = back_inserter(samples);
  for (int i = 0; i < 10; ++i)
    *sample_i++ = normal_dist();

  EXPECT_EQ(samples.size(), 10);
  for (const auto& s : samples)
    EXPECT_NE(s, 0.f);
}

TEST(TestNormalDistribution, test_normal_image)
{
  auto sizes = (Vector2i::Ones() * 5).eval();
  const auto image = normal(sizes);

  EXPECT_MATRIX_EQ(image.sizes(), sizes);
  for (int i = 0; i < image.size(); ++i)
    EXPECT_NE(image.data()[i], 0.f);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
