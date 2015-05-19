// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <gtest/gtest.h>
#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/MultiArray.hpp>

using namespace DO;
using namespace std;

TEST(DO_SARA_Core_Test, eigenExtensionTest)
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

  for (int i = 0; i < m.rows(); ++i)
    for (int j = 0; j < m.cols(); ++j)
      EXPECT_EQ(m(i,j), a);

  for (int i = 0; i < n.rows(); ++i)
    for (int j = 0; j < n.cols(); ++j)
      EXPECT_EQ(n(i,j), b);


  // Double that matrix
  m.array() += n.array();
  // Check that matrix
  for (int i = 0; i < m.rows(); ++i)
    for (int j = 0; j < m.cols(); ++j)
      EXPECT_EQ(m(i,j), (a+b).eval());

  EXPECT_EQ(m(0,0)*n(0,0), (a+b)*b);

  // Double that matrix
  m.array() *= n.array();
  // Check that matrix
  for (int i = 0; i < m.rows(); ++i)
    for (int j = 0; j < m.cols(); ++j)
      EXPECT_EQ(m(i,j), (a+b)*b);

  m.matrix() += n.matrix();
  (m.array() * n.array()) + n.array() / m.array();
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
