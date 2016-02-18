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

#include <gtest/gtest.h>

#include <DO/Sara/Geometry/Tools/Polynomial.hpp>
#include <DO/Sara/Geometry/Tools/PolynomialRoots.hpp>


using namespace std;
using namespace DO::Sara;


TEST(TestPolynomialRoots, test_quadratic_polynomial_roots)
{
  Polynomial<double,2> P{-1, 0, 2};

  bool real_roots;
  complex<double> x1, x2;
  roots(P, x1, x2, real_roots);

  EXPECT_TRUE(real_roots);
  EXPECT_NEAR(abs(P(x1)), 0., 1e-10);
  EXPECT_NEAR(abs(P(x2)), 0., 1e-10);
}

TEST(TestPolynomialRoots, test_cubic_polynomial_roots)
{
  Polynomial<double, 3> P{-6, 11, -6, 1};

  // Roots are 1, 2 and 3.
  complex<double> x1, x2, x3;
  roots(P, x1, x2, x3);

  const double eps = 1e-9;

  EXPECT_NEAR(abs(P(x1)), 0., eps);
  EXPECT_NEAR(abs(P(x2)), 0., eps);
  EXPECT_NEAR(abs(P(x3)), 0., eps);
}

TEST(TestPolynomialRoots, test_quartic_polynomial_roots)
{
  // Roots are 1, 2, 3 and 4.
  Polynomial<double, 4> P{24, -50, 35, -10, 1};

  complex<double> x1, x2, x3, x4;
  roots(P, x1, x2, x3, x4);

  const double eps = 1e-10;

  EXPECT_NEAR(abs(P(x1)), 0., eps);
  EXPECT_NEAR(abs(P(x2)), 0., eps);
  EXPECT_NEAR(abs(P(x3)), 0., eps);
  EXPECT_NEAR(abs(P(x4)), 0., eps);
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
