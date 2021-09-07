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
  Univariate::UnivariatePolynomial<double, 2> P{-1., 0., 2.};

  bool real_roots;
  complex<double> x1, x2;
  roots(P, x1, x2, real_roots);

  BOOST_CHECK(real_roots);
  BOOST_CHECK_SMALL(abs(P(x1)), 1e-10);
  BOOST_CHECK_SMALL(abs(P(x2)), 1e-10);
}

BOOST_AUTO_TEST_CASE(test_cubic_polynomial_roots)
{
  Univariate::UnivariatePolynomial<double, 3> P{-6., 11., -6., 1.};

  // Roots are 1, 2 and 3.
  complex<double> x1, x2, x3;
  roots(P, x1, x2, x3);

  const double eps = 1e-9;

  BOOST_CHECK_SMALL(abs(P(x1)), eps);
  BOOST_CHECK_SMALL(abs(P(x2)), eps);
  BOOST_CHECK_SMALL(abs(P(x3)), eps);
}

BOOST_AUTO_TEST_CASE(test_quartic_polynomial_roots)
{
  // Roots are 1, 2, 3 and 4.
  Univariate::UnivariatePolynomial<double, 4> P{24., -50., 35., -10., 1.};

  complex<double> x1, x2, x3, x4;
  roots(P, x1, x2, x3, x4);

  const double eps = 1e-10;

  BOOST_CHECK_SMALL(abs(P(x1)), eps);
  BOOST_CHECK_SMALL(abs(P(x2)), eps);
  BOOST_CHECK_SMALL(abs(P(x3)), eps);
  BOOST_CHECK_SMALL(abs(P(x4)), eps);
}

BOOST_AUTO_TEST_SUITE_END()
