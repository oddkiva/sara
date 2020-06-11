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

#define BOOST_TEST_MODULE "Halide Backend/Binary Operators"

#include <boost/test/unit_test.hpp>

#include <Eigen/Dense>

#include <drafts/Halide/Components/TinyLinearAlgebra.hpp>


using namespace Halide;

using Mat2 = DO::Sara::HalideBackend::Matrix<2, 2>;


template <int M, int N>
auto eval(const DO::Sara::HalideBackend::Matrix<M, N>& expr)
{
  auto fn = Halide::Func{};
  auto x = Halide::Var{"x"};
  fn(x) = expr;

  auto r = fn.realize(std::vector<int32_t>{1});

  auto m = Eigen::Matrix<float, M, N>{};
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < M; ++i)
      m(i, j) = Halide::Buffer<float>(r[i * M + j])(0);

  return m;
}

auto eval(const Halide::Expr& expr)
{
  auto fn = Halide::Func{};
  auto x = Halide::Var{"x"};
  fn(x) = expr;

  auto r = fn.realize(std::vector<int32_t>{1});

  return Halide::Buffer<float>(r)(0);
}

BOOST_AUTO_TEST_CASE(test_matrix)
{
  auto a = Mat2{};
  a(0, 0) = 2.1f; a(0, 1) = 0.f;
  a(1, 0) = 0.f; a(1, 1) = 1.f;

  auto b = Mat2{};
  b(0, 0) = 2.f; b(0, 1) = 3.f;
  b(1, 0) = 1.5f; b(1, 1) = 1.f;

  auto c = a * b;
  c = c + c;

  std::cout << eval(a) << std::endl;
  std::cout << eval(b) << std::endl;
  std::cout << eval(c) << std::endl;
  std::cout << eval(c * DO::Sara::HalideBackend::inverse(c)) << std::endl;

  std::cout << "trace(c) = " << eval(trace(c)) << std::endl;
  std::cout << "det(c) = " << eval(det(c)) << std::endl;
}
