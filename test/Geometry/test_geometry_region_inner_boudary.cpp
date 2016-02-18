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

#include <DO/Sara/Geometry/Algorithms.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestRegionInnerBoundary, test_compute_region_inner_boundary)
{
  auto regions = Image<int>{ 5, 5 };
  regions.matrix() <<
    0, 0, 1, 2, 3,
    0, 1, 2, 2, 3,
    0, 2, 2, 2, 2,
    4, 4, 2, 2, 2,
    4, 4, 2, 2, 5;

  auto true_boundaries = vector<vector<Point2i>>{
    { Point2i{ 0, 2 }, Point2i{ 0, 1 }, Point2i{ 0, 0 }, Point2i{ 1, 0 } },
    { Point2i{ 2, 0 }, Point2i{ 1, 1 } },
    { Point2i{ 3, 0 }, Point2i{ 2, 1 }, Point2i{ 1, 2 }, Point2i{ 2, 3 },
      Point2i{ 2, 4 }, Point2i{ 3, 4 }, Point2i{ 4, 3 }, Point2i{ 4, 2 },
      Point2i{ 3, 1 } },
    { Point2i{ 4, 0 }, Point2i{ 4, 1 } },
    { Point2i{ 0, 3 }, Point2i{ 1, 3 }, Point2i{ 0, 4 }, Point2i{ 1, 4 } },
    { Point2i{ 4, 4 } }
  };

  auto actual_boundaries = compute_region_inner_boundaries(regions);

  EXPECT_EQ(true_boundaries.size(), actual_boundaries.size());
  for (size_t i = 0; i < true_boundaries.size(); ++i)
    ASSERT_ITEMS_EQ(true_boundaries[i], actual_boundaries[i]);
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
