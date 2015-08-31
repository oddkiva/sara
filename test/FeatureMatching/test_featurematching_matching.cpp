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

#include <DO/Sara/FeatureMatching.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestFeatureMatching, test_ann_matching)
{
  Set<OERegion, RealDescriptor> keys1, keys2;

  keys1.resize(1, 2);
  keys1.features[0].coords() = Point2f::Zero();
  keys1.descriptors[0] = Vector2f::Zero();

  keys2.resize(10, 2);
  for (int i = 0; i < 10; ++i)
  {
    keys2.features[i].coords() = Point2f::Zero()*float(i);
    keys2.descriptors[i] = Vector2f::Ones()*float(i);
  }

  AnnMatcher matcher{ keys1, keys2, 0.6f };
  auto matches = matcher.compute_matches();

  EXPECT_EQ(matches.size(), 1);

  const auto& m = matches.front();
  EXPECT_EQ(m.x(), keys1.features[0]);
  EXPECT_EQ(m.y(), keys2.features[0]);
  EXPECT_EQ(m.score(), 0.f);

}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}