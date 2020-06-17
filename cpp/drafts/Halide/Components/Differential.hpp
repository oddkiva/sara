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

#include <drafts/Halide/Components/TinyLinearAlgebra.hpp>


namespace DO::Sara::HalideBackend {

  using namespace Halide;

  template <typename Input>
  auto gradient(const Input& in,               //
                const Expr& x, const Expr& y)  //
      -> Vector<2>
  {
    auto g = Vector<2>{};
    g(0) = (in(x + 1, y) - in(x - 1, y)) / 2;
    g(1) = (in(x, y + 1) - in(x, y - 1)) / 2;
    return g;
  }

  template <typename Input>
  auto hessian(const Input& in,               //
               const Expr& x, const Expr& y)  //
      -> Matrix<2, 2>
  {
    auto dxx = in(x + 1, y) + in(x - 1, y) - 2 * in(x, y);
    auto dyy = in(x, y + 1) + in(x, y - 1) - 2 * in(x, y);

    auto dxy = (in(x + 1, y + 1) - in(x - 1, y - 1) -  //
                in(x + 1, y - 1) + in(x - 1, y - 1)) /
               4;

    auto h = Matrix2{};
    h(0, 0) = dxx; h(0, 1) = dxy;
    h(1, 0) = dxy; h(1, 1) = dyy;

    return h;
  }

  template <typename Input>
  auto laplacian(const Input& in,               //
                 const Expr& x, const Expr& y)  //
      -> Expr
  {
    return in(x + 1, y) + in(x - 1, y) - 2 * in(x, y);
  }


  template <typename Input>
  auto scale_space_gradient(const Input& in,                              //
                            const Expr& x, const Expr& y, const Expr& s)  //
      -> Vector<3>
  {
    auto g = Vector<3>{};
    g(0) = (in(x + 1, y, s) - in(x - 1, y, s)) / 2;
    g(1) = (in(x, y + 1, s) - in(x, y - 1, s)) / 2;
    g(2) = (in(x, y, s + 1) - in(x, y, s - 1)) / 2;
    return g;
  }

  template <typename Input>
  auto scale_space_hessian(const Input& in,                              //
                           const Expr& x, const Expr& y, const Expr& s)  //
      -> Matrix<3, 3>
  {
    Expr dxx = in(x + 1, y, s) + in(x - 1, y, s) - 2 * in(x, y, s);
    Expr dyy = in(x, y + 1, s) + in(x, y - 1, s) - 2 * in(x, y, s);
    Expr dss = in(x, y, s + 1) + in(x, y, s - 1) - 2 * in(x, y, s);

    Expr dxy = (in(x + 1, y + 1, s) - in(x - 1, y - 1, s) -  //
                in(x + 1, y - 1, s) + in(x - 1, y - 1, s)) /
               4;
    Expr dxs = (in(x + 1, y, s) - in(x - 1, y, s) -  //
                in(x + 1, y, s - 1) + in(x - 1, y, s - 1)) /
               4;
    Expr dys = (in(x, y + 1, s) - in(x, y - 1, s) -  //
                in(x, y + 1, s - 1) + in(x, y - 1, s)) /
               4;

    auto h = Matrix3{};
    h(0, 0) = dxx; h(0, 1) = dxy; h(0, 2) = dxs;
    h(1, 0) = dxy; h(1, 1) = dyy; h(1, 2) = dys;
    h(2, 0) = dxs; h(2, 1) = dys; h(2, 2) = dss;

    return h;
  }

}  // namespace DO::Sara::HalideBackend
