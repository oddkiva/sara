// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Core/Usual Functions"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Math/UsualFunctions.hpp>


namespace sara = DO::Sara;


BOOST_AUTO_TEST_SUITE(TestUsualFunctions)

BOOST_AUTO_TEST_CASE(test_square)
{
  {
    constexpr auto x = 10.12;
    constexpr auto y = sara::square(x);
    BOOST_CHECK_EQUAL(y, x * x);
  }

  {
    const auto x = 13.12;
    const auto y = sara::square(x);
    BOOST_CHECK_EQUAL(y, x * x);
  }
}

BOOST_AUTO_TEST_CASE(test_cubic)
{
  {
    constexpr auto x = 10.12;
    constexpr auto y = sara::cubic(x);
    BOOST_CHECK_EQUAL(y, x * x * x);
  }

  {
    const auto x = 13.12;
    const auto y = sara::cubic(x);
    BOOST_CHECK_EQUAL(y, x * x * x);
  }
}

BOOST_AUTO_TEST_CASE(test_quartic)
{
  {
    constexpr auto x = 10.12;
    constexpr auto y = sara::quartic(x);
    BOOST_CHECK_EQUAL(y, (x * x) * (x * x));
  }

  {
    const auto x = 13.12;
    const auto y = sara::quartic(x);
    BOOST_CHECK_EQUAL(y, (x * x) * (x * x));
  }
}

BOOST_AUTO_TEST_SUITE_END()
