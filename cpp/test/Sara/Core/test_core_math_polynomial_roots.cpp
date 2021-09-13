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

#define BOOST_TEST_MODULE "Geometry/Tools/Polynomial Roots"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Math/PolynomialRoots.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestPolynomialRoots)

BOOST_AUTO_TEST_CASE(test_quadratic_polynomial_roots)
{
  const auto P = UnivariatePolynomial<double, 2>{-1., 0., 2.};

  auto x1 = complex<double>{};
  auto x2 = complex<double>{};
  roots(P, x1, x2);

  BOOST_CHECK_SMALL(abs(P(x1)), 1e-10);
  BOOST_CHECK_SMALL(abs(P(x2)), 1e-10);

  double x1r, x2r;
  const auto roots_are_real = compute_quadratic_real_roots(P, x1r, x2r);
  BOOST_CHECK(roots_are_real);
  BOOST_CHECK_SMALL(abs(P(x1r)), 1e-10);
  BOOST_CHECK_SMALL(abs(P(x2r)), 1e-10);
}

BOOST_AUTO_TEST_CASE(test_cubic_polynomial_roots)
{
  const auto P = UnivariatePolynomial<double, 3>{-6., 11., -6., 1.};

  // Roots are 1, 2 and 3.
  auto x1 = complex<double>{};
  auto x2 = complex<double>{};
  auto x3 = complex<double>{};
  roots(P, x1, x2, x3);

  const auto eps = 1e-9;

  BOOST_CHECK_SMALL(abs(P(x1)), eps);
  BOOST_CHECK_SMALL(abs(P(x2)), eps);
  BOOST_CHECK_SMALL(abs(P(x3)), eps);

  auto x1r = double{};
  auto x2r = double{};
  auto x3r = double{};
  const auto roots_are_real = compute_cubic_real_roots(P, x1r, x2r, x3r);
  BOOST_CHECK(roots_are_real);
  BOOST_CHECK_SMALL(abs(P(x1r)), eps);
  BOOST_CHECK_SMALL(abs(P(x2r)), eps);
  BOOST_CHECK_SMALL(abs(P(x3r)), eps);
}

BOOST_AUTO_TEST_CASE(test_quartic_polynomial_roots)
{
  // Roots are 1, 2, 3 and 4.
  const auto P = UnivariatePolynomial<double, 4>{24., -50., 35., -10., 1.};

  auto x1 = complex<double>{};
  auto x2 = complex<double>{};
  auto x3 = complex<double>{};
  auto x4 = complex<double>{};
  roots(P, x1, x2, x3, x4);

  const auto eps = 1e-10;

  BOOST_CHECK_SMALL(abs(P(x1)), eps);
  BOOST_CHECK_SMALL(abs(P(x2)), eps);
  BOOST_CHECK_SMALL(abs(P(x3)), eps);
  BOOST_CHECK_SMALL(abs(P(x4)), eps);
}

BOOST_AUTO_TEST_SUITE_END()
