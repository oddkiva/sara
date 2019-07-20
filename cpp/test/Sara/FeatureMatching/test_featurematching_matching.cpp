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

#define BOOST_TEST_MODULE                                                      \
  "FeatureMatching/Approximate Nearest Neighbor-Based Matching"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/FeatureMatching.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestFeatureMatching)

BOOST_AUTO_TEST_CASE(test_ann_matching)
{
  auto keys1 = KeypointList<OERegion, float>{};
  auto keys2 = KeypointList<OERegion, float>{};

  resize(keys1, 1, 2);
  auto& [f1, v1] = keys1;
  f1[0].coords = Point2f::Zero();
  v1[0].row_vector() = RowVector2f::Zero();

  resize(keys2, 10, 2);
  auto& [f2, v2] = keys2;
  for (auto i = 0; i < size(keys2); ++i)
  {
    f2[i].coords = Point2f::Ones() * float(i);
    v2[i].row_vector() = RowVector2f::Ones() * float(i);
  }

  AnnMatcher matcher{keys1, keys2, 0.6f};
  auto matches = matcher.compute_matches();

  BOOST_CHECK_EQUAL(1u, matches.size());

  const auto& m = matches.front();
  BOOST_CHECK_EQUAL(f1[0], m.x());
  BOOST_CHECK_EQUAL(f2[0], m.y());
  BOOST_CHECK_EQUAL(0.f, m.score());
}

BOOST_AUTO_TEST_SUITE_END()
