// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Core/Math/Univariate Polynomial"

#include <DO/Sara/Core/Math/UnivariatePolynomial.hpp>

#include <boost/test/unit_test.hpp>

#include <iostream>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestPolynomialCalculus)

BOOST_AUTO_TEST_CASE(test)
{
  auto P = (Z - 2.) * (Z - 2.) * (Z + 3.);
  cout << "P = " << P.to_string() << std::endl;

  auto Q = Z + 3.;
  cout << "Q = " << Q.to_string() << std::endl;

  auto res = P / Q;
  cout << "Euclidean division P/Q" << endl;
  cout << "Quotient = " << res.first.to_string() << std::endl;
  cout << "Remainder = " << res.second.to_string() << std::endl;

  cout << (res.first * Q + res.second).to_string() << std::endl;

  cout << "P(2) = " << P(2) << endl;
  cout << "P(-3) = " << P(-3) << endl;
  cout << "P(-3.0000001) = " << P(-3.0000001) << endl;
  cout << "P(2.02) = " << P(2.02) << endl;
  cout << "P(2.1) = " << P(2.1) << endl;
}

BOOST_AUTO_TEST_SUITE_END()
