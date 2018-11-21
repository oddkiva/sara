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

#define BOOST_TEST_MODULE "Core/Math/Polynomial"

#include <DO/Sara/Core/Math/Polynomial.hpp>

#include <boost/test/unit_test.hpp>

#include <iostream>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestPolynomialCalculus)

BOOST_AUTO_TEST_CASE(test_variable)
{
  auto x = Variable{"x"};
  auto y = Variable{"y"};

  BOOST_CHECK(x < y);
}

BOOST_AUTO_TEST_CASE(test_monomial)
{
  auto x = Monomial{Variable{"x"}};
  auto y = Monomial{Variable{"y"}};
  auto z = Monomial{Variable{"z"}};

  auto x3 = x.pow(3);
  auto x2 = x.pow(2);
  auto xy = x * y;
  auto xy2z3 = x * y.pow(2) * z.pow(3);

  std::cout << x.to_string() << std::endl;
  std::cout << y.to_string() << std::endl;
  std::cout << xy.to_string() << std::endl;
  std::cout << x2.to_string() << std::endl;
  std::cout << x3.to_string() << std::endl;
  std::cout << xy2z3.to_string() << std::endl;
}

BOOST_AUTO_TEST_CASE(test_polynomial)
{
  auto x = Monomial{Variable{"x"}};
  auto y = Monomial{Variable{"y"}};
  auto z = Monomial{Variable{"z"}};
  auto one = Monomial{Variable{}};

  std::cout << one.to_string() << std::endl;

  Matrix3d X, Y, Z, W;
  Polynomial<Matrix3d> E;
  E.coeffs[x] = X;
  E.coeffs[y] = Y;
  E.coeffs[z] = Z;
  E.coeffs[one] = W;

  const auto EEt = E * E.t();

  auto P1 = EEt * E;
  auto P2 = trace(EEt) * E;

  auto P = P1 + P2;

  auto Q = det(E);
}

BOOST_AUTO_TEST_SUITE_END()
