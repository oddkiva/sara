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

#include <iomanip>

#include <boost/test/unit_test.hpp>

#include <Eigen/Dense>

#include <DO/Shakti/Halide/Components/Differential.hpp>
#include <DO/Shakti/Halide/Components/Evaluation.hpp>


using DO::Shakti::HalideBackend::eval;
using DO::Shakti::HalideBackend::Matrix2;
using DO::Shakti::HalideBackend::Matrix3;

using namespace Halide;


BOOST_AUTO_TEST_CASE(test_gradient)
{
  auto x = Var{"x"}, y = Var{"y"};
  auto f = Func{};
  f(x, y) = cos(x) * sin(y);

  auto df = Func{};
  df(x, y) = DO::Shakti::HalideBackend::gradient(f, x, y);

  const auto r = df.realize({3, 3});
  const auto df_dx = Buffer<float>(r[0]);
  const auto df_dy = Buffer<float>(r[1]);

  std::cout << "Gradient =" << std::endl;
  for (auto v = 0; v < 3; ++v)
  {
    for (auto u = 0; u < 3; ++u)
      std::cout << "[" << df_dx(u, v) << ", " << df_dy(u, v) << "]  ";
    std::cout << std::endl;
  }
}

BOOST_AUTO_TEST_CASE(test_laplacian)
{
  auto x = Var{"x"}, y = Var{"y"};
  auto f = Func{};
  f(x, y) = cos(x) * sin(y);

  auto del_f = Func{};
  del_f(x, y) = DO::Shakti::HalideBackend::laplacian(f, x, y);

  const auto r = Buffer<float>(del_f.realize({3, 3}));

  std::cout << "Laplacian =" << std::endl;
  for (auto v = 0; v < 3; ++v)
  {
    for (auto u = 0; u < 3; ++u)
      std::cout << r(u, v) << " ";
    std::cout << std::endl;
  }
}

BOOST_AUTO_TEST_CASE(test_hessian)
{
  auto x = Var{"x"}, y = Var{"y"};
  auto f = Func{};
  f(x, y) = cos(x) * sin(y);

  auto df = Func{};
  df(x, y) = DO::Shakti::HalideBackend::hessian(f, x, y);

  const auto r = df.realize({3, 3});
  const auto h = std::array<Buffer<float>, 3>{r[0], r[1], r[2]};

  std::cout << "Hessian =" << std::endl;
  for (auto v = 0; v < 3; ++v)
  {
    for (auto u = 0; u < 3; ++u)
      std::cout << "["                                                     //
                << h[0](u, v) << ", " << h[1](u, v) << ", " << h[2](u, v)  //
                << "]  ";
    std::cout << std::endl;
  }
}

// BOOST_AUTO_TEST_CASE(test_scale_space_gradient)
// {
//   auto x = Var{"x"}, y = Var{"y"}, s = Var{"s"};
//   auto f = Func{};
//   f(x, y, s) = cos(x) * sin(y) * exp(-s);
//
//   auto df = Func{};
//   df(x, y, s) = DO::Shakti::HalideBackend::scale_space_gradient(f, x, y, s);
//
//   const auto r = df.realize({3, 3, 3});
//   auto g = std::array<Buffer<float>, 3>{};
//   for (int i = 0; i < 3; ++i)
//     g[i] = r[i];
//
//   std::cout << "Scale-space gradient =" << std::endl;
//   for (auto v = 0; v < 3; ++v)
//   {
//     for (auto u = 0; u < 3; ++u)
//       std::cout << "["                                                     //
//                 << g[0](u, v) << ", " << g[1](u, v) << ", " << g[2](u, v)  //
//                 << "]  ";
//     std::cout << std::endl;
//   }
// }

// BOOST_AUTO_TEST_CASE(test_scale_space_hessian)
// {
//   auto x = Var{"x"}, y = Var{"y"}, s = Var{"s"};
//   auto f = Func{};
//   f(x, y, s) = cos(x) * sin(y) * exp(-s);
//
//   auto d2f = Func{};
//   d2f(x, y, s) = DO::Shakti::HalideBackend::scale_space_hessian(f, x, y, s);
//
//   const auto r = d2f.realize({3, 3, 3});
//   auto h = std::array<Buffer<float>, 9>{};
//   for (int i = 0; i < 9; ++i)
//     h[i] = r[i];
//
//   std::cout << "Scale-space hessian =" << std::endl;
//   for (auto v = 0; v < 3; ++v)
//   {
//     for (auto u = 0; u < 3; ++u)
//       std::cout                                                              //
//           << std::setprecision(3) << "["                                     //
//           << h[0](u, v) << ", " << h[1](u, v) << ", " << h[2](u, v) << ", "  //
//           << h[3](u, v) << ", " << h[4](u, v) << ", " << h[5](u, v) << ", "  //
//           << h[6](u, v) << ", " << h[7](u, v) << ", " << h[8](u, v) << "]  ";
//     std::cout << std::endl;
//   }
// }
