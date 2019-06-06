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
  auto x = Monomial{variable("x")};
  auto y = Monomial{variable("y")};
  auto z = Monomial{variable("z")};

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
