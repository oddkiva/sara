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
  auto x = Monomial{variable("x")};
  auto y = Monomial{variable("y")};
  auto z = Monomial{variable("z")};
  auto one_ = Monomial{one()};

  Matrix3d X, Y, Z, W;
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

  Polynomial<Matrix3d> E;
  E.coeffs[x] = X;
  E.coeffs[y] = Y;
  E.coeffs[z] = Z;
  E.coeffs[one_] = W;

  const auto EEt = E * E.t();

  auto P1 = EEt * E;
  auto P2 = trace(EEt) * E;
  P2 *= -0.5;

  auto P = P1 + P2;

  const auto P00 = P(0, 0);
  std::cout << "P00 has " << P00.coeffs.size() << " monomials" << std::endl;
  for (const auto& c: P00.coeffs)
    std::cout << "Monomial: " << c.first.to_string() << std::endl;

  auto Q = det(E);
  std::cout << "det(E) = " << Q.to_string() << std::endl;

  auto index = [](char c) { return std::size_t(c - 'a'); };
  auto letter = [](int c) { return char('a' + c); };
  const auto num_equations = index('m') + 1;

  // As per Nister paper.
  auto A = std::vector<Polynomial<double>>{num_equations};
  A[index('a')] = det(E);
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      const auto c = index('b') + 3 * i + j;
      A[c] = P(i, j);
    }
  }

  A[index('k')] = A[index('e')] - A[index('f')] * z;
  A[index('l')] = A[index('g')] - A[index('h')] * z;
  A[index('m')] = A[index('i')] - A[index('j')] * z;

  for (auto i = 0u; i < A.size(); ++i)
    std::cout << "A[" << letter(i) << "] = " << A[i].to_string() << std::endl;


  // Calculate <n> = det(B)
  // 1. Perform Gauss-Jordan elimination on A and stop four rows earlier.
  //    lower diagonal of A is zero (minus some block)
  // 2. B is the right-bottom block after Gauss-Jordan elimination of A.
  // 3. [x, y, 1]^T is a non-zero null vector in Null(B).
  // 4. Therefore solve polynomial in z: det(B) = 0.
  // 5. Use Sturm-sequences to bracket the root.
  // 6. Recover x = p1(z) / p3(z), y = p2(z) / p3(z)
  //    where p1, p2, p3 are obtained from expansion by minors in det(B).
  // 7. to solve det(B) = 0, use sturm sequence to extract roots and polish the
  //    roots.
  // 8. Recover R and t from E.
  //    E ~ U diag(1, 1, 0) V
  //    t = [U(0,2), U(1,2), U(2, 2)]
  //    R = U * D * V.transpose() or R = U * D.transpose() * V.transpose();
  // 9. 4 possible second camera matrices.
  //    Check cheirality.
}

BOOST_AUTO_TEST_SUITE_END()
