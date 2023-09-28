// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Geometry/Algorithms/Robust Estimation"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Random.hpp>

#include "AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_random_samples)
{
  const auto S = random_samples(10, 2, 10);
  SARA_DEBUG << "S =\n" << S.matrix() << std::endl;
}
