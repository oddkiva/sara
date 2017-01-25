// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <gtest/gtest.h>

#include <DO/Sara/FeatureMatching/KeyProximity.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestKeyProximity, test_computations)
{
  auto key_proximity = KeyProximity{};
  auto f1 = OERegion{ Point2f{ 0.f, 0.f }, 1.f };
  auto f2 = OERegion{ Point2f{ 0.f, 0.1f }, 1.1f };

  EXPECT_MATRIX_EQ(
    key_proximity.mapped_squared_metric(f1).covariance_matrix(),
    Matrix2f::Identity());

  EXPECT_TRUE(key_proximity(f1, f2));
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}