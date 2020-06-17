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
    Expr dxx = in(x + 1, y) + in(x - 1, y) - 2 * in(x, y);
    Expr dyy = in(x, y + 1) + in(x, y - 1) - 2 * in(x, y);

    Expr dxy = (in(x + 1, y + 1) - in(x - 1, y - 1) -  //
                in(x + 1, y - 1) + in(x - 1, y - 1)) /
               4;
    return {dxx, dxy,  //
            dxy, dyy};
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
    return {(in(x + 1, y, s) - in(x - 1, y, s)) / 2,
            (in(x, y + 1, s) - in(x, y - 1, s)) / 2,
            (in(x, y, s + 1) - in(x, y, s - 1)) / 2};
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

    return {dxx, dxy, dxs,  //
            dxy, dyy, dys,  //
            dxs, dys, dss};
  }

}  // namespace DO::Sara::HalideBackend
