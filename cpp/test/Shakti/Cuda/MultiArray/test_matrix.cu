// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Shakti/MultiArray/Matrix"

#include <boost/test/unit_test.hpp>

#include <DO/Shakti/Cuda/MultiArray.hpp>


using namespace std;
using namespace DO::Shakti;


BOOST_AUTO_TEST_CASE(test_constructor_1d)
{
  Vector1i p{10};
  BOOST_CHECK_EQUAL(10, p);

  int& p_as_scalar = p;
  const int& p_as_const_scalar = p;
  BOOST_CHECK_EQUAL(10, p_as_scalar);
  BOOST_CHECK_EQUAL(10, p_as_const_scalar);

  p_as_scalar = 12;
  BOOST_CHECK_EQUAL(12, p_as_scalar);
  BOOST_CHECK_EQUAL(12, p_as_const_scalar);
}

BOOST_AUTO_TEST_CASE(test_constructor_2d)
{
  Vector2i p{2, 1};
  BOOST_CHECK_EQUAL(p.x(), 2);
  BOOST_CHECK_EQUAL(p.y(), 1);

  const int sizes[] = {1, 3};
  p = sizes;
  BOOST_CHECK_EQUAL(p.x(), sizes[0]);
  BOOST_CHECK_EQUAL(p.y(), sizes[1]);
}
