// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
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
  for (size_t i = 0; i < keys2.size(); ++i)
  {
    keys2.features[i].coords() = Point2f::Ones()*float(i);
    keys2.descriptors[i] = Vector2f::Ones()*float(i);
  }

  AnnMatcher matcher{ keys1, keys2, 0.6f };
  auto matches = matcher.compute_matches();

  EXPECT_EQ(size_t{ 1 }, matches.size());

  const auto& m = matches.front();
  EXPECT_EQ(keys1.features[0], m.x());
  EXPECT_EQ(keys2.features[0], m.y());
  EXPECT_EQ(0.f, m.score());
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
