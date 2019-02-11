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
#include <DO/Sara/Core/Math/JenkinsTraub.hpp>

#include <boost/test/unit_test.hpp>

#include <iostream>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestUnivariatePolynomial)

BOOST_AUTO_TEST_CASE(test_polynomial_arithmetics)
{
  auto P = (Z - 2.) * (Z - 2.) * (Z + 3.);
  //cout << "P = " << P.to_string() << endl;

  auto Q = Z + 3.;
  //cout << "Q = " << Q.to_string() << endl;

  auto res = P / Q;
  //cout << "Euclidean division P/Q" << endl;
  //cout << "Quotient = " << res.first.to_string() << endl;
  //cout << "Remainder = " << res.second.to_string() << endl;

  //cout << (res.first * Q + res.second).to_string() << endl;

  //cout << "P(2) = " << P(2) << endl;
  //cout << "P(-3) = " << P(-3) << endl;
  //cout << "P(-3.0000001) = " << P(-3.0000001) << endl;
  //cout << "P(2.02) = " << P(2.02) << endl;
  //cout << "P(2.1) = " << P(2.1) << endl;

  // Everything is OK here.
}


BOOST_AUTO_TEST_CASE(test_jenkins_traub_sigma)
{
  auto s1 = std::complex<double>{2, -1};
  auto sigma = sigma_(s1);
  cout << "sigma = " << sigma << endl;
  cout << "sigma(s1) = " << sigma(s1)  << endl;
}


BOOST_AUTO_TEST_CASE(test_jenkins_traub_stage_1)
{
  auto P = (Z - 2.) * (Z - 2.) * (Z + 3.);
  cout << "P = " << P << endl;

  auto K0 = K0_(P);
  cout << "K0 = " << K0 << endl;

  auto s1 = std::complex<double>(0.1);
  auto s2 = std::conj(s1);
  auto sigma = sigma_(s1);

  auto K1 = K1_(K0, P, sigma, s1, s2);

//  // Root approximation.
//  auto t = [](double s, const UnivariatePolynomial<double>& P,
//              const UnivariatePolynomial<double>& K) -> double {
//    return s - P(s) / K(s);
//  };
//
//  auto K0_ = K0(P);
//  BOOST_ASSERT(K0_.degree() == 2);
//  cout << "K0.degree() = " << K0_.degree() << endl;
//  cout << "K0 = " << K0_ << endl;
//  cout << "t(s, P, K0) = " << t(s, P, K0_) << endl;
//
//  auto K1_ = K1_stage1(P, K0_, s);
//  BOOST_ASSERT(K1_.degree() <= K0_.degree());
//  BOOST_ASSERT(K1_.degree() == 2);
//
//  cout << "K1 = " << K1_ << endl;
//  cout << "t(s, P, K1) = " << t(s, P, K1_) << endl;
//
//  for (int i = 0; i < 4; ++i)
//  {
//  }
}



BOOST_AUTO_TEST_SUITE_END()
