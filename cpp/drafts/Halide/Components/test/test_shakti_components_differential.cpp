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

#define BOOST_TEST_MODULE "Halide Backend/Differential"

#include <boost/test/unit_test.hpp>

#include <Eigen/Dense>

#include <drafts/Halide/Components/Differential.hpp>
#include <drafts/Halide/Components/Evaluation.hpp>


using DO::Sara::HalideBackend::Matrix2;
using DO::Sara::HalideBackend::Matrix3;
using DO::Sara::HalideBackend::eval;

using namespace Halide;


BOOST_AUTO_TEST_CASE(test_gradient)
{
  auto x = Var{"x"}, y = Var{"y"};
  auto f = Func{};
  f(x, y) = cos(x) * sin(y);

  auto df = Func{};
  df(x, y)  = DO::Sara::HalideBackend::gradient(f, x, y);

  auto r = df.realize({3, 3});
  auto df_dx = Buffer<float>(r[0]);
  auto df_dy = Buffer<float>(r[1]);

  for (auto v = 0; v < 3; ++v)
  {
    for (auto u = 0; u < 3; ++u)
      std::cout << "[" << df_dx(u, v) << ", " << df_dy(u, v) << "]  ";
    std::cout << std::endl;
  }
}
