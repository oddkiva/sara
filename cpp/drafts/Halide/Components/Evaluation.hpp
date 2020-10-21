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

#pragma once

#include <drafts/Halide/Components/TinyLinearAlgebra.hpp>

#include <Eigen/Core>


namespace DO::Shakti::HalideBackend {

  template <int M, int N>
  auto eval(const Matrix<M, N>& expr)
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

}  // namespace DO::Shakti::HalideBackend
