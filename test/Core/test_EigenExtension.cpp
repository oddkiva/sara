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
#include <DO/Core/EigenExtension.hpp>
#include <DO/Core/MultiArray.hpp>

using namespace DO;
using namespace std;

TEST(DO_Core_Test, eigenExtensionTest)
{
  typedef Matrix2f SuperScalar;
  typedef MultiArray<SuperScalar, 2> Mat2i;
  SuperScalar a; a << 1, 2, 3, 4;
  SuperScalar b; b << 1, 1, 2, 3;

  Mat2i m(2,2);
  Mat2i n(2,2);

  // Initialize the matrices m and n.
  m.array().fill(a);
  n.array().fill(b);
  
  // Check m
  cout << "Check m" << endl;
  for (int i = 0; i < m.rows(); ++i)
    for (int j = 0; j < m.cols(); ++j)
      //cout << "m(" << i << "," << j << ") = " << endl << m(i,j) << endl;
      EXPECT_EQ(m(i,j), a);
  // Check n
  cout << "Check n" << endl;
  for (int i = 0; i < n.rows(); ++i)
    for (int j = 0; j < n.cols(); ++j)
      //cout << "n(" << i << "," << j << ") = " << endl << n(i,j) << endl;
      EXPECT_EQ(n(i,j), b);


  // Double that matrix
  cout << "m.array() += n.array()" << endl;
  m.array() += n.array();
  // Check that matrix
  for (int i = 0; i < m.rows(); ++i)
    for (int j = 0; j < m.cols(); ++j)
      //cout << "m(" << i << "," << j << ") = " << endl << m(i,j) << endl;
      EXPECT_EQ(m(i,j), (a+b).eval());

  cout << "m(0,0)*n(0,0)=" << endl;
  cout << m(0,0)*n(0,0) << endl;
  EXPECT_EQ(m(0,0)*n(0,0), (a+b)*b);

  // Double that matrix
  cout << "m.array() *= n.array()" << endl;
  m.array() *= n.array();
  // Check that matrix
  for (int i = 0; i < m.rows(); ++i)
    for (int j = 0; j < m.cols(); ++j)
      //cout << "m(" << i << "," << j << ") = " << endl << m(i,j) << endl;
      EXPECT_EQ(m(i,j), (a+b)*b);

  m.matrix() += n.matrix();
  (m.array() * n.array()) + n.array() / m.array();
}

int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}