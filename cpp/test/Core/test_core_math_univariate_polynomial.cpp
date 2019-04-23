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

#include <DO/Sara/Core/Math/JenkinsTraub.hpp>
#include <DO/Sara/Core/Math/NewtonRaphson.hpp>
#include <DO/Sara/Core/Math/UnivariatePolynomial.hpp>

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <iomanip>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestUnivariatePolynomial)

BOOST_AUTO_TEST_CASE(test_polynomial_arithmetics)
{
  auto P = (Z - 2.) * (Z - 2.) * (Z + 3.);
  //cout << "P = " << P << endl;

  auto Q = Z + 3.;
  //cout << "Q = " << Q << endl;

  auto res = P / Q;
  //cout << "Euclidean division P/Q" << endl;
  //cout << "Quotient = " << res.first << endl;
  //cout << "Remainder = " << res.second << endl;
  //cout << (res.first * Q + res.second) << endl;
  BOOST_CHECK_EQUAL(res.first.degree(), 2);
  BOOST_CHECK_EQUAL(res.second.degree(), 0);

  BOOST_CHECK_EQUAL((P / 2)[3], 0.5);

  BOOST_CHECK_EQUAL(P(0), 12);
  BOOST_CHECK_EQUAL(Q(0), 3);

  //cout << "P(2) = " << P(2) << endl;
  BOOST_CHECK_CLOSE(P(2), 0, std::numeric_limits<double>::epsilon());

  //cout << "P(-3) = " << P(-3) << endl;
  BOOST_CHECK_CLOSE(P(-3), 0, std::numeric_limits<double>::epsilon());

  //cout << "P(-3.0000001) = " << P(-3.0000001) << endl;
  BOOST_CHECK_LE(std::abs(P(-3.0000001)), 1e-4);

  //cout << "P(2.02) = " << P(2.02) << endl;

  //cout << "P(2.1) = " << P(2.1) << endl;
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(TestJenkinsTraub)

BOOST_AUTO_TEST_CASE(test_newton_raphson)
{
  auto P = (Z - 2.) * (Z - 2.) * (Z + 3.);
  //cout << "P = " << P << endl;

  auto z = 1.1;
  auto newton_raphson = NewtonRaphson<double>{P};
  z = newton_raphson(z, 100);
  //cout << setprecision(12) << std::abs(z - 2) << endl;

  BOOST_CHECK_CLOSE(z, 2, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_compute_moduli_lower_bound)
{
  {
    auto P = 20. * (Z + 1.07) * (Z - 2.) * (Z + 3.);
    auto x = compute_moduli_lower_bound(P);
    BOOST_CHECK_LE(x, 1.07);
  }

  {
    auto P = 20. * (Z + 1.07) * (Z - 2.) * (Z + 3.) * (Z + 0.6);
    auto x = compute_moduli_lower_bound(P);
    BOOST_CHECK_LE(x, 0.6);
  }
}

BOOST_AUTO_TEST_CASE(test_jenkins_traub_stage_1)
{
  auto P = (Z - 2.) * (Z - 2.) * (Z + 3.);
  cout << "P = " << P << endl;
  cout << "P(0) = " << P(0) << endl;

  JenkinsTraub rpoly{P};
  rpoly.stage1();

  rpoly.stage2();
  rpoly.stage3();
}

BOOST_AUTO_TEST_SUITE_END()
