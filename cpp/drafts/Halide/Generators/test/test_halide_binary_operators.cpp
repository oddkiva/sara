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

#define BOOST_TEST_MODULE "Halide Backend/Binary Operators"

#include <boost/test/unit_test.hpp>

#include <drafts/Halide/Utilities.hpp>
#include <drafts/Halide/BinaryOperators.hpp>


using namespace Halide;


BOOST_AUTO_TEST_CASE(test_gaussian_blur)
{
  using namespace DO::Sara;

  auto a = Image<float>{8, 8};
  auto b = Image<float>{8, 8};
  auto res = Image<float>{8, 8};
  auto expected = Image<float>{8, 8};

  a.flat_array().fill(2);
  b.flat_array().fill(1);
  expected.flat_array().fill(1);

  DO::Shakti::HalideBackend::subtract(a, b, res);

  BOOST_CHECK_EQUAL(res.matrix(), expected.matrix());
}
