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

  auto E = x * X + y * Y + z * Z + one_ * W;

  const auto EEt = E * E.t();

  auto P1 = EEt * E;
  auto P2 = trace(EEt) * E;
  P2 *= -0.5;
  auto P = P1 + P2;

#ifdef DEBUG
  const auto P00 = P(0, 0);
  std::cout << "P00 has " << P00.coeffs.size() << " monomials" << std::endl;
  for (const auto& c: P00.coeffs)
    std::cout << "Monomial: " << c.first.to_string() << std::endl;
#endif

  auto Q = det(E);
#ifdef DEBUG
  std::cout << "det(E) = " << Q.to_string() << std::endl;
#endif

  const Monomial monomials[] = {x.pow(3), y.pow(3), x.pow(2) * y, x * y.pow(2),
                                x.pow(2) * z, x.pow(2), y.pow(2) * z, y.pow(2),
                                x * y * z, x * y,
                                //
                                x, x * z, x * z.pow(2),
                                //
                                y, y * z, y * z.pow(2),
                                //
                                one_, z, z.pow(2), z.pow(3)};

  // ===========================================================================
  // As per Nister paper.
  //
  Matrix<double, 10, 20> A;
  A.setZero();

  // Save Q in the matrix.
  for (int j = 0; j < 20; ++j)
  {
    auto coeff = Q.coeffs.find(monomials[j]);
    if (coeff == Q.coeffs.end())
      continue;
    A(0, j) = coeff->second;
  }

  // Save P in the matrix.
  for (int a = 0; a < 3; ++a)
  {
    for (int b = 0; b < 3; ++b)
    {
      const auto i = 3 * a + b;
      for (int j = 0; j < 20; ++j)
        A(i, j) = P(a, b).coeffs[monomials[j]];
    }
  }
  cout << "A =\n" << A << endl;


  // ===========================================================================
  // 1. Perform Gauss-Jordan elimination on A and stop four rows earlier.
  //    lower diagonal of A is zero (minus some block)
  Eigen::FullPivLU<Matrix<double, 10, 20>> lu(A);
  Matrix<double, 10, 20> U = lu.matrixLU().triangularView<Upper>();
  cout << "U =\n" << U << endl;

  // Calculate <n> = det(B)
  // 2. B is the right-bottom block after Gauss-Jordan elimination of A.
  Matrix<double, 3, 10> B;
  B.setZero();
  RowVectorXd e = A.row(int('e' - 'a'));
  RowVectorXd f = A.row(int('f' - 'a'));
  RowVectorXd g = A.row(int('g' - 'a'));
  RowVectorXd h = A.row(int('h' - 'a'));
  RowVectorXd i = A.row(int('i' - 'a'));
  RowVectorXd j = A.row(int('j' - 'a'));

  auto to_poly = [&monomials](const auto& row_vector) {
    auto p = Polynomial<double>{};
    for (int i = 0; i < row_vector.size(); ++i)
      p.coeffs[monomials[i]] = row_vector[i];
    return p;
  };

  auto k = to_poly(e); auto k2 = (z * to_poly(f)); k = k - k2;
  //auto l = to_poly(g.eval()) - z * to_poly(h.eval());
  //auto m = to_poly(i.eval()) - z * to_poly(j.eval());


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
