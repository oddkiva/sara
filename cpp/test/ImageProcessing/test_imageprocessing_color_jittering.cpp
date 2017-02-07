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

#include <DO/Sara/ImageProcessing/ColorJitter.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestNormalDistribution, test_sampling)
{
  auto dist = NormalDistribution{};

  auto samples = vector<float>{};
  auto sample_i = back_inserter(samples);
  for (int i = 0; i < 10; ++i)
    *sample_i++ = dist();

  EXPECT_EQ(samples.size(), 10);
  for (const auto& s : samples)
    EXPECT_NE(s, 0.f);
}

TEST(TestNormalDistribution, test_randn_vector)
{
  const auto randn = NormalDistribution{};
  auto v = Vector3f{};
  randn(v);
  EXPECT_TRUE(v != Vector3f::Zero());

  auto v2 = MatrixXf{3, 1};
  randn(v2);
  EXPECT_TRUE(v2 != Vector3f::Zero());
}

TEST(TestNormalDistribution, test_randn_image)
{
  const auto randn = NormalDistribution{};
  auto image = Image<Rgb32f>{5, 5};
  image.array().fill(Rgb32f::Zero());

  randn(image);
  for (const auto& pixel : image)
    EXPECT_TRUE(pixel != Vector3f::Zero());
}

TEST(TestNormalDistribution, test_add_randn_noise)
{
  auto image = Image<Rgb32f>{5, 5};
  image.array().fill(Rgb32f::Zero());

  const auto randn = NormalDistribution{};
  add_randn_noise(image, 0.5, randn);
  for (const auto& pixel : image)
    EXPECT_TRUE(pixel != Vector3f::Zero());
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
