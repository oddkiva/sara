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

#include <drafts/Halide/Components/Differential.hpp>
#include <drafts/Halide/Components/LocalExtremum.hpp>


namespace DO { namespace Shakti { namespace HalideBackend {

  inline auto sign(const Halide::Expr& val)
  {
    return Halide::select(val > 0,                          //
                          +1,                               //
                          Halide::select(val < 0, -1, 0));  //
  }

  template <typename Input>
  auto on_edge(const Input& in, const Halide::Expr& edge_ratio,  //
               const Halide::Var& x, const Halide::Var& y)       //
  {
    const auto h = hessian(in, x, y);
    return pow(trace(h), 2) * edge_ratio >=
           pow(1 + edge_ratio, 2) * abs(det(h));
  }

  template <typename Input>
  auto on_edge(const Input& in, const Halide::Expr& edge_ratio,  //
               const Halide::Var& x, const Halide::Var& y,       //
               const Halide::Var& c, const Halide::Var& n)       //
  {
    const auto h = hessian(in, x, y, c, n);
    return pow(trace(h), 2) * edge_ratio >=
           pow(1 + edge_ratio, 2) * abs(det(h));
  }


  auto residual(const Matrix<3, 3>& hessian, const Vector<3>& gradient)
      -> Vector<3>
  {
    return -(inverse(hessian) * gradient);
  }


  //! @brief Find the local extrema in Halide.
  template <typename Input>
  auto is_dog_extremum(                                                    //
      const Input& prev, const Input& curr, const Input& next,             //
      const Halide::Expr& edge_ratio, const Halide::Expr& extremum_thres,  //
      const Halide::Var& x, const Halide::Var& y)                          //
  {
    auto is_max = local_scale_space_max(prev, curr, next, x, y);
    auto is_min = local_scale_space_min(prev, curr, next, x, y);
    auto is_strong = abs(curr(x, y)) > 0.8f * extremum_thres;
    auto is_not_on_edge = !on_edge(curr, edge_ratio, x, y);

    return Halide::select(
        is_max && is_strong && is_not_on_edge,                           //
        Halide::cast<std::int8_t>(1) /* good local max! */,              //
        Halide::select(                                                  //
            is_min && is_strong && is_not_on_edge,                       //
            Halide::cast<std::int8_t>(-1) /* good local min! */,         //
            Halide::cast<std::int8_t>(0) /* not a local extremum! */));  //
  }

  template <typename Input>
  auto is_dog_extremum(                                                    //
      const Input& prev, const Input& curr, const Input& next,             //
      const Halide::Expr& edge_ratio, const Halide::Expr& extremum_thres,  //
      const Halide::Var& x, const Halide::Var& y,                          //
      const Halide::Var& c, const Halide::Var& n)                          //
  {
    auto is_max = local_scale_space_max(prev, curr, next, x, y, c, n);
    auto is_min = local_scale_space_min(prev, curr, next, x, y, c, n);
    auto is_strong = abs(curr(x, y, c, n)) > 0.8f * extremum_thres;
    auto is_not_on_edge = !on_edge(curr, edge_ratio, x, y, c, n);

    return Halide::select(
        is_max && is_strong && is_not_on_edge,                           //
        Halide::cast<std::int8_t>(1) /* good local max! */,              //
        Halide::select(                                                  //
            is_min && is_strong && is_not_on_edge,                       //
            Halide::cast<std::int8_t>(-1) /* good local min! */,         //
            Halide::cast<std::int8_t>(0) /* not a local extremum! */));  //
  }


  // Count the number of extrema in Halide.
  template <typename Input>
  auto count_extrema(const Input& in,                               //
                     const Halide::Expr& w, const Halide::Expr& h)  //
  {
    auto r = Halide::RDom(0, w, 0, h);
    return Halide::sum(Halide::abs(in(r.x, r.y)));
  }


  template <typename Input>
  auto refine_extremum_v1(const Input& I0, const Input& I1, const Input& I2,  //
                       const Halide::Expr& x, const Halide::Expr& y)       //
      -> Tuple
  {
    auto h = scale_space_hessian(I0, I1, I2, x, y);
    auto g = scale_space_gradient(I0, I1, I2, x, y);

    auto res = residual(h, g);

    auto new_value = I1(x, y) + 0.5f * dot(g, res);
    auto success = Halide::abs(res(0)) < 1.5f &&                    //
                   Halide::abs(res(1)) < 1.5f &&                    //
                   Halide::abs(I1(x, y)) < Halide::abs(new_value);  //

    return {res(0), res(1), res(2),  //
            new_value,               //
            success};                //
  }

  template <typename Input>
  auto refine_extremum_v1(const Input& I0, const Input& I1, const Input& I2,  //
                          const Halide::Expr& x, const Halide::Expr& y,       //
                          const Halide::Var& c, const Halide::Var& n)         //
      -> Tuple
  {
    auto h = scale_space_hessian(I0, I1, I2, x, y, c, n);
    auto g = scale_space_gradient(I0, I1, I2, x, y, c, n);

    auto res = residual(h, g);

    auto new_value = I1(x, y, c, n) + 0.5f * dot(g, res);
    auto success = Halide::abs(res(0)) < 1.5f &&                          //
                   Halide::abs(res(1)) < 1.5f &&                          //
                   Halide::abs(I1(x, y, c, n)) < Halide::abs(new_value);  //

    return {res(0), res(1), res(2),  //
            new_value,               //
            success};                //
  }


  template <typename Input>
  auto refine_extremum_v2(const Input& I0, const Input& I1, const Input& I2,  //
                          const Halide::Expr& x, const Halide::Expr& y)       //
      -> Tuple
  {
    auto h = scale_space_hessian(I0, I1, I2, x, y);
    auto g = scale_space_gradient(I0, I1, I2, x, y);

    auto res = residual(h, g);
    auto success = Halide::abs(res(0)) < 1.5f || Halide::abs(res(1)) < 1.5f;

    // Shift to the best neighboring pixel for further precision.
    auto shift_x = Halide::select(Halide::abs(res(0)) < 0.5f,  //
                                  0, sign(res(0)));
    auto shift_y = Halide::select(Halide::abs(res(1)) < 0.5f,  //
                                  0, sign(res(1)));          //

    auto x1 = x + shift_x;
    auto y1 = y + shift_y;

    h = scale_space_hessian(I0, I1, I2, x1, y1);
    g = scale_space_gradient(I0, I1, I2, x1, y1);
    res = residual(h, g);

    auto new_value = I1(x1, y1) + 0.5f * dot(g, res);
    success = Halide::abs(I1(x1, y1)) < Halide::abs(new_value);

    return {res(0), res(1), res(2), new_value, success};
  }

}}}  // namespace DO::Sara::HalideBackend
