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
  // cout << "P = " << P << endl;

  auto z = 1.1;
  auto newton_raphson = NewtonRaphson<double>{P};
  z = newton_raphson(z, 100, numeric_limits<double>::epsilon());
  // cout << setprecision(12) << std::abs(z - 2) << endl;

  BOOST_CHECK_CLOSE(z, 2, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_compute_root_moduli_lower_bound)
{
  {
    auto P = 20. * (Z + 1.07) * (Z - 2.) * (Z + 3.);
    auto x = compute_root_moduli_lower_bound(P);
    BOOST_CHECK(0 < x && x < 1.07);
  }

  {
    auto P = 20. * (Z + 1.07) * (Z - 2.) * (Z + 3.) * (Z + 0.6);
    auto x = compute_root_moduli_lower_bound(P);
    BOOST_CHECK(0 < x && x < 0.6);
  }
}

BOOST_AUTO_TEST_CASE(test_jenkins_traub_1)
{
  auto P = (Z - 0.000000123) *      //
           (Z - 0.0000001230001) *  //
           (Z - 0.00199999) *       //
           (Z - 0.0021) *           //
           (Z - 0.0537) *           //
           (Z - 0.2) *              //
           (Z - 1.0000001) *        //
           (Z - 2.) *               //
           (Z + 3.) *               //
           (Z - 10.);

  const auto roots = rpoly(P);

  for (const auto& root: roots)
    cout << "P(" << setprecision(12) << root << ") = " << setprecision(12)
         << P(root) << std::endl;
}

BOOST_AUTO_TEST_CASE(test_jenkins_traub_2)
{
  auto P = Z.pow<double>(7)             //
           - 6.01 * Z.pow<double>(6)    //
           + 12.54 * Z.pow<double>(5)   //
           - 8.545 * Z.pow<double>(4)   //
           - 5.505 * Z.pow<double>(3)   //
           + 12.545 * Z.pow<double>(2)  //
           - 8.035 * Z                  //
           + 2.01;

  const auto roots = rpoly(P);

  for (const auto& root: roots)
    cout << "P(" << setprecision(12) << root << ") = " << setprecision(12)
         << P(root) << std::endl;
}

BOOST_AUTO_TEST_CASE(test_jenkins_traub_3)
{
  auto P = (Z - 1.23e-4) *  //
           (Z - 1.23e-1) *  //
           (Z - 1.23e+2) *  //
           (Z - 1.23e+5);

  const auto roots = rpoly(P);

  for (const auto& root: roots)
  {
    cout << "P(" << setprecision(12) << root << ") = " << setprecision(12)
         << P(root) << std::endl;
    P = (P / (Z - root.real())).first;
  }
}

BOOST_AUTO_TEST_CASE(test_jenkins_traub_4)
{
  auto P =                                        //
      (5.576312106019016) * Z.pow<double>(10) +   //
      (18.62488243410351) * Z.pow<double>(9) +    //
      (-105.89974320540716) * Z.pow<double>(8) +  //
      (-562.5089223272787) * Z.pow<double>(7) +   //
      (-692.3579951742573) * Z.pow<double>(6) +   //
      (1909.2968243041091) * Z.pow<double>(5) +   //
      (11474.831044766257) * Z.pow<double>(4) +   //
      (25844.743360023815) * Z.pow<double>(3) +   //
      (20409.84088622263) * Z.pow<double>(2) +    //
      (-28342.548095425875) * Z +                 //
      (-52412.8655144021);

  const auto roots = rpoly(P);

  for (const auto& root: roots)
  {
    cout << "P(" << setprecision(12) << root << ") = " << setprecision(12)
         << P(root) << std::endl;
  }
}

BOOST_AUTO_TEST_SUITE_END()
