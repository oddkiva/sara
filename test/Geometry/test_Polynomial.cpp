// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <gtest/gtest.h>
#include <DO/Geometry/Tools/Polynomial.hpp>
#include <DO/Geometry/Tools/PolynomialRoots.hpp>
#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

TEST(DO_Geometry_Test, testQuadraticPolynomialRoots)
{
  bool verbose = false;

  double coeff[3] = { -1.0, 0.0, 2.0 };
  Polynomial<double,2> P(coeff);
 
  bool realRoots;
  complex<double> x1, x2;
  roots(P, x1, x2, realRoots);
  
  if (verbose)
  {
    cout << P << endl;
    cout << "x1 = " << x1 << " and x2 = " << x2 << endl;
    cout << "P(" << x1 << ") = " << P(x1) << endl;
    cout << "P(" << x2 << ") = " << P(x2) << endl;
    cout << endl;
  }

  EXPECT_TRUE(realRoots);
  EXPECT_NEAR(abs(P(x1)), 0., std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(abs(P(x2)), 0., std::numeric_limits<double>::epsilon());
}

TEST(DO_Geometry_Test, testCubicPolynomialRoots)
{
  bool verbose = false;

  // check quadratic equation solver
  complex<double> x1, x2, x3;
  for(int i = 0; i < 10; ++i)
  {
    double p[4] ={
      static_cast<double>(rand()%10),
      static_cast<double>(rand()%10),
      static_cast<double>(rand()%10),
      static_cast<double>(1+rand()%9)
    };
    Polynomial<double, 3> P(p);
    roots(P, x1, x2, x3);

    if (verbose)
    {
      cout << "iteration " << i << endl;
      cout << "x1 = " << x1 << " and x2 = " << x2 << " and x3 = " << x3 << endl;
      cout << "|P(" << x1 << ")| = " << abs(P(x1)) << endl;
      cout << "|P(" << x2 << ")| = " << abs(P(x2)) << endl;
      cout << "|P(" << x3 << ")| = " << abs(P(x3)) << endl;
      cout << endl;
    }

    const double eps = 1e-10;

    EXPECT_NEAR(abs(P(x1)), 0., eps);
    EXPECT_NEAR(abs(P(x2)), 0., eps);
    EXPECT_NEAR(abs(P(x3)), 0., eps);
  }
}

TEST(DO_Geometry_Test, testQuarticPolynomialRoots)
{
  bool verbose = false;

  // check quadratic equation solver
  complex<double> x1, x2, x3, x4;
  for(int i = 0; i < 10; ++i)
  {
    double p[5] = {
      static_cast<double>(rand()%100000),
      static_cast<double>(rand()%10),
      static_cast<double>(rand()%10),
      static_cast<double>(rand()%10),
      static_cast<double>(rand()%10)+1
    };
    Polynomial<double, 4> P(p);
    roots(P, x1, x2, x3, x4);

    if (verbose)
    {
      cout << "iteration " << i << endl;
      cout << "x1 = " << x1 << " and x2 = " << x2 << endl;
      cout << "x3 = " << x3 << " and x4 = " << x4 << endl;
      cout << "|P(" << x1 << ")| = " << abs(P(x1)) << endl;
      cout << "|P(" << x2 << ")| = " << abs(P(x2)) << endl;
      cout << "|P(" << x3 << ")| = " << abs(P(x3)) << endl;
      cout << "|P(" << x4 << ")| = " << abs(P(x4)) << endl;
      cout << endl;
    }

    const double eps = 1e-10;

    EXPECT_NEAR(abs(P(x1)), 0., eps);
    EXPECT_NEAR(abs(P(x2)), 0., eps);
    EXPECT_NEAR(abs(P(x3)), 0., eps);
    EXPECT_NEAR(abs(P(x4)), 0., eps);
  }
}

int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}