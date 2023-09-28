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

BOOST_AUTO_TEST_CASE(test_simple_linear_algebra)
{
  auto a = Matrix3f{};
  a(0, 0) = 2; a(0, 1) = 0; a(0, 2) = 0;
  a(1, 0) = 0; a(1, 1) = 1; a(1, 2) = 0;
  a(2, 0) = 0; a(2, 1) = 0; a(2, 2) = 1;

  std::cout << a << std::endl;

  BOOST_CHECK_EQUAL(det(a), 2);
  BOOST_CHECK_EQUAL(trace(a), 4);

  auto b = Matrix3f{};
  b(0, 0) = 0.5; b(0, 1) = 0; b(0, 2) = 0;
  b(1, 0) = 0.0; b(1, 1) = 1; b(1, 2) = 0;
  b(2, 0) = 0.0; b(2, 1) = 0; b(2, 2) = 1;

  BOOST_CHECK_EQUAL(inverse(a), b);
  BOOST_CHECK_EQUAL(inverse(a) * a, Matrix3f::Identity());

  std::cout << b << std::endl;

  const auto c = Vector3f::Ones();

  const auto d = -(b * c);
  BOOST_CHECK_EQUAL(d, Vector3f(-0.5, -1, -1));
  std::cout << d << std::endl;
}
