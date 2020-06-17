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

#include <drafts/Halide/Components/Differential.hpp>
#include <drafts/Halide/MyHalide.hpp>


namespace DO::Sara::HalideBackend {

  using namespace Halide;


  template <typename Input>
  auto on_edge(const Input& in, const Expr& edge_ratio,   //
               const Var& x, const Var& y, const Var& n)  //
  {
    const auto h = hessian(in, x, y, n);
    return pow(trace(h), 2) * edge_ratio >=
           pow(1 + edge_ratio, 2) * abs(determinant(h));
  }

  template <typename Input>
  auto residual(const Matrix<3, 3>& hessian, const Matrix<3, 1>& gradient)
      -> Vector<3>
  {
    return -(inverse(hessian) * gradient);
  }

  // Find the local extrema in Halide.
  template <typename Input>
  auto is_dog_extremum(                                         //
      const Input& prev, const Input& curr, const Input& next,  //
      const Expr& edge_ratio, const Expr& extremum_thres,       //
      const Var& x, const Var& y, const Var& n)                 //
      -> Expr
  {
    return (local_scale_space_max(prev, curr, next, x, y, n) ||
            local_scale_space_min(prev, curr, next, x, y, n)) &&
           !on_edge(curr, edge_ratio, x, y, n) &&
           abs(curr(x, y)) > 0.8f * extremum_thres;
  }

  // Count the number of extrema in Halide.
  template <typename Input>
  auto count_extrema(const Input& in, const Expr& w, const Expr& h) -> int
  {
    auto r = RDom(0, w, 0, h);
    return sum(in(r));
  }

  // We have collected the list of extrema. Refine the extrema.
  template <typename Input>
  auto refine_extremum(                                             //
      const Input& I, const Expr& x, const Expr& y, const Expr& s,  //
      Matrix<3, 1>& p, Expr& success)                               //
      -> void
  {
    auto h = scale_space_hessian(I, x, y, s);
    auto g = scale_space_gradient(I, x, y, s);
    auto res = residual(h, g);

    success = abs(res(0)) < 1.5f || abs(res(1)) < 1.5f;

    // Shift to the best neighboring pixel for further precision.
    auto shift_x = select(h(0) > 0, 1, -1);
    auto shift_y = select(h(1) > 0, 1, -1);
    p(0) = select(abs(res(0)) > 0.5f, p(0) + shift_x, p(0));
    p(1) = select(abs(res(1)) > 0.5f, p(1) + shift_y, p(1));

    h = scale_space_hessian(I, x, y, s);
    g = scale_space_gradient(I, x, y, s);
    res = residual(h, g);

    auto new_value = I(x, y, s) + 0.5f * res;

    success = abs(I(x, y, s)) < new_value;
    p = select(success, p + res, p);
  }


}  // namespace DO::Sara::HalideBackend
