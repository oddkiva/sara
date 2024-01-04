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

#include <DO/Shakti/Halide/Utilities.hpp>
#include <DO/Shakti/Halide/LocalExtrema.hpp>


using namespace Halide;


BOOST_AUTO_TEST_CASE(test_gaussian_blur)
{
  using namespace DO::Sara;

  auto a = Image<float>{3, 3};
  auto b = Image<float>{3, 3};
  auto c = Image<float>{3, 3};
  auto res = Image<std::uint8_t>{3, 3};
  auto expected = Image<std::uint8_t>{3, 3};

  a.flat_array().fill(0);
  b.flat_array().fill(0);
  c.flat_array().fill(0);

  b(1, 1) = 10;

  res.flat_array().fill(0);

  expected.flat_array().fill(0);
  expected(1, 1) = 1;

  DO::Shakti::HalideBackend::local_max(a, b, c, res);

  BOOST_CHECK_EQUAL(res.matrix(), expected.matrix());
}
