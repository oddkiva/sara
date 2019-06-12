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

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Core/Math/Polynomial.hpp>
#include <DO/Sara/Core/Math/JenkinsTraub.hpp>

#include <boost/test/unit_test.hpp>

#include <iostream>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestPolynomialCalculus)

BOOST_AUTO_TEST_CASE(test_variable)
{
  auto x = variable("x");
  auto y = variable("y");

  BOOST_CHECK(x < y);
}

BOOST_AUTO_TEST_CASE(test_monomial)
{
  const auto x_ = variable("x");
  const auto y_ = variable("y");
  const auto z_ = variable("z");

  const auto x = Monomial{x_};
  const auto y = Monomial{y_};
  const auto z = Monomial{z_};

  const auto x3 = x.pow(3);
  const auto x2 = x.pow(2);
  const auto xy = x * y;
  const auto xy2z3 = x * y.pow(2) * z.pow(3);

  std::cout << x.to_string() << std::endl;
  std::cout << y.to_string() << std::endl;
  std::cout << xy.to_string() << std::endl;
  std::cout << x2.to_string() << std::endl;
  std::cout << x3.to_string() << std::endl;
  std::cout << xy2z3.to_string() << std::endl;

  const auto x_eval = x.eval<double>({{x_, 1.}});
  BOOST_CHECK_EQUAL(x_eval, 1.);

  const auto xy2z3_eval = xy2z3.eval<double>({{x_, 1.}, {y_, 2.}, {z_, 3.}});
  BOOST_CHECK_EQUAL(xy2z3_eval, 1 * 4 * 27);
}

BOOST_AUTO_TEST_CASE(test_polynomial_multiplication)
{
  const auto x_ = variable("x");
  const auto y_ = variable("y");
  const auto z_ = variable("z");

  const auto x = Monomial{x_};
  const auto y = Monomial{y_};
  const auto z = Monomial{z_};

  {
    const auto P = (1. * x + 1. * y);
    const auto Q = (1. * x - 1. * y);
    const auto PQ = (1. * x.pow(2) - 1. * y.pow(2));
    BOOST_CHECK(P * Q ==  PQ);
    for (const auto& c: PQ.coeffs)
      cout << c.first.to_string() << " " << c.second << endl;
    std::cout << "P = " << P.to_string() << std::endl;
    std::cout << "Q = " << Q.to_string() << std::endl;
    std::cout << "P * Q = " << (P * Q).to_string() << std::endl;
    std::cout << "PQ = " << PQ.to_string() << std::endl;
    std::cout << "PQ - P*Q = " << (PQ - P * Q).to_string() << std::endl;
  }

  {
    const auto P = (1. * x.pow(3) + 1. * x.pow(2));
    const auto Q = (1. * y.pow(2) + 0.5 * z - 1. * x * y);
    const auto PQ = P * Q;
    std::cout << "P = " << P.to_string() << std::endl;
    std::cout << "Q = " << Q.to_string() << std::endl;
    std::cout << "P * Q = " << (P * Q).to_string() << std::endl;
    /*
     * x^3 y^2 + 0.5 x^3 z - x^4 y    4 + 0.5*2
     * x^2 y^2 + 0.5 x^2 z - x^3 y
     */
    const auto PQ_eval = PQ.eval<double>({{x_, 1.}, {y_, 2.}, {z_, 1}});
    BOOST_CHECK_EQUAL(PQ_eval, 2. * (4. + 0.5 * 1 - 1. * 2.));
  }
}

BOOST_AUTO_TEST_CASE(test_polynomial)
{
  std::array<Matrix3d, 4> null_space_bases;
  auto& [X, Y, Z, W] = null_space_bases;
  X << 1, 0, 0,
       0, 0, 0,
       0, 0, 0;

  Y << 0, 1, 0,
       0, 0, 0,
       0, 0, 0;

  Z << 0, 0, 1,
       0, 0, 0,
       0, 0, 0;

  W << 0, 0, 0,
       1, 0, 0,
       0, 0, 0;

  const auto x = Monomial{variable("x")};
  const auto y = Monomial{variable("y")};
  const auto z = Monomial{variable("z")};
  const auto one_ = Monomial{one()};

  const auto E = x * X + y * Y + z * Z + one_ * W;

  const auto EEt = E * E.t();

  const auto P = det(E);
  const auto Q = EEt * E - 0.5 * trace(EEt) * E;
}

BOOST_AUTO_TEST_SUITE_END()
