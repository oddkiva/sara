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

  auto solver = NisterFivePointAlgorithm{};
  const auto E = solver.essential_matrix_expression(null_space_bases);

  const auto A = solver.build_epipolar_constraints(E);
  cout << "A =\n" << A << endl;

//  // 4. Therefore solve polynomial in z: det(B) = 0.
//  // 5. Use Sturm-sequences to bracket the root.
//  // 6. Recover x = p1(z) / p3(z), y = p2(z) / p3(z)
//  //    where p1, p2, p3 are obtained from expansion by minors in det(B).
//  // 7. to solve det(B) = 0, use sturm sequence to extract roots and polish the
//  //    roots.
//  // 8. Recover R and t from E.
//  //    E ~ U diag(1, 1, 0) V
//  //    t = [U(0,2), U(1,2), U(2, 2)]
//  //    R = U * D * V.transpose() or R = U * D.transpose() * V.transpose();
//  // 9. 4 possible second camera matrices.
//  //    Check cheirality.
}

BOOST_AUTO_TEST_SUITE_END()
