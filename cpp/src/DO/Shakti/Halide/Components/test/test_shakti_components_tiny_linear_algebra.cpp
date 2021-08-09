// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Halide Backend/Tiny Linear Algebra"

#include <boost/test/unit_test.hpp>

#include <Eigen/Dense>

#include <DO/Shakti/Halide/Components/TinyLinearAlgebra.hpp>
#include <DO/Shakti/Halide/Components/Evaluation.hpp>


using DO::Shakti::HalideBackend::Matrix2;
using DO::Shakti::HalideBackend::Matrix3;
using DO::Shakti::HalideBackend::eval;

using namespace Halide;


BOOST_AUTO_TEST_CASE(test_matrix_set_zero)
{
  auto a = Matrix2{};
  a.set_zero();
  BOOST_CHECK_EQUAL(eval(a), Eigen::Matrix2f::Zero().eval());
}

BOOST_AUTO_TEST_CASE(test_matrix2_operations)
{
  auto a = Matrix2{};
  a(0, 0) = 2.1f;
  a(0, 1) = 0.f;
  a(1, 0) = 0.f;
  a(1, 1) = 1.f;
  BOOST_CHECK_EQUAL(eval(det(a)), 2.1f);
  BOOST_CHECK_EQUAL(eval(trace(a)), 3.1f);

  auto b = Matrix2{};
  b(0, 0) = 2.f;
  b(0, 1) = 3.f;
  b(1, 0) = 1.5f;
  b(1, 1) = 1.f;

  auto c = a * b;
  c = c + c;

  BOOST_CHECK_EQUAL(eval(c), eval(2.f * (a * b)));
  std::cout << eval(2.f * (a * b)) << std::endl;

  using DO::Shakti::HalideBackend::inverse;
  BOOST_CHECK_EQUAL(eval(c * inverse(c)), Eigen::Matrix2f::Identity());
  std::cout << eval(c * inverse(c)) << std::endl;
}

BOOST_AUTO_TEST_CASE(test_matrix3_operations)
{
  auto a = Matrix3{};
  a.set_zero();
  a(0, 0) = 2.1f;
  a(1, 1) = 1.f;
  a(2, 2) = 3.f;
  BOOST_CHECK_CLOSE(eval(det(a)), 6.3f, 1e-5f);
  BOOST_CHECK_CLOSE(eval(trace(a)), 6.1f, 1e-6f);

  using DO::Shakti::HalideBackend::inverse;
  std::cout << eval(a * inverse(a)) << std::endl;
  std::cout << eval(inverse(a)) << std::endl;

  auto b = Matrix3{};
  b(0, 0) =  0.f; b(0, 1) = -3.f; b(0, 2) = -2.f;
  b(1, 0) = +1.f; b(1, 1) = -4.f; b(1, 2) = -2.f;
  b(2, 0) = -3.f; b(2, 1) = +4.f; b(2, 2) = +1.f;

  auto true_inv_b = Eigen::Matrix3f{};
  true_inv_b  <<
    +4.f, -5.f, -2.f,
    +5.f, -6.f, -2.f,
    -8.f, +9.f, +3.f;
  BOOST_CHECK_EQUAL(eval(inverse(b)), true_inv_b);

  BOOST_CHECK_EQUAL(eval(b * inverse(b)), Eigen::Matrix3f::Identity().eval());
  std::cout << eval(inverse(b)) << std::endl;
  std::cout << eval(b * inverse(b)) << std::endl;
}
