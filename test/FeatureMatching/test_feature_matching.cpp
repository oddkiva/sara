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


using namespace std;
using namespace DO::Sara;


TEST(TestFeatureMatching, test_ann_matching)
{
  Set<OERegion, RealDescriptor> keys1, keys2;

  AnnMatcher matcher{ keys1, keys2, 0.6f };
  //auto matches = matcher.compute_matches();

  // TODO.
}

int main(int argc, char **argv)
{
  return 0;
}