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

#define BOOST_TEST_MODULE "FeatureMatching/Key Proximity Predicate"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/FeatureMatching/KeyProximity.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestKeyProximity)

BOOST_AUTO_TEST_CASE(test_computations)
{
  auto key_proximity_predicate = KeyProximity{};
  auto f1 = OERegion{Point2f{0.f, 0.f}, 1.f};
  auto f2 = OERegion{Point2f{0.f, 0.1f}, 1.1f};

  BOOST_CHECK_EQUAL(
      key_proximity_predicate.mapped_squared_metric(f1).covariance_matrix(),
      Matrix2f::Identity());

  BOOST_CHECK(key_proximity_predicate(f1, f2));
}

BOOST_AUTO_TEST_SUITE_END()
