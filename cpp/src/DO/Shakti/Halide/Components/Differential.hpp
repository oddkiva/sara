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

#include <DO/Shakti/Halide/Components/TinyLinearAlgebra.hpp>


namespace DO::Shakti::HalideBackend {

  //! @brief Nice API.
  //! @{
  template <typename Input>
  auto gradient(const Input& in,                               //
                const Halide::Expr& x, const Halide::Expr& y)  //
      -> Vector<2>
  {
    auto g = Vector<2>{};
    g(0) = (in(x + 1, y) - in(x - 1, y)) / 2;
    g(1) = (in(x, y + 1) - in(x, y - 1)) / 2;
    return g;
  }

  template <typename Input>
  auto hessian(const Input& in,                               //
               const Halide::Expr& x, const Halide::Expr& y)  //
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
  auto laplacian(const Input& in,                               //
                 const Halide::Expr& x, const Halide::Expr& y)  //
      -> Halide::Expr
  {
    return in(x + 1, y) + in(x - 1, y) - 2 * in(x, y);
  }

  template <typename Input>
  auto gradient_3d(const Input& f,  //
                   const Halide::Expr& x,   //
                   const Halide::Expr& y,   //
                   const Halide::Expr& z,   //
                   const Halide::Expr& n)   //
      -> Vector<3>
  {
    auto g = Vector<3>{};
    g(0) = (f(x + 1, y, z, n) - f(x - 1, y, z, n)) / 2;
    g(1) = (f(x, y + 1, z, n) - f(x, y - 1, z, n)) / 2;
    g(2) = (f(x, y, z + 1, n) - f(x, y, z - 1, n)) / 2;
    return g;
  }

  template <typename Input>
  auto hessian_3d(const Input& f,  //
                  const Halide::Expr& x,   //
                  const Halide::Expr& y,   //
                  const Halide::Expr& z,   //
                  const Halide::Expr& n)   //
      -> Matrix<3, 3>
  {
    using Halide::Expr;

    Expr dxx = f(x + 1, y, z, n) - 2 * f(x, y, z, n) + f(x - 1, y, z, n);
    Expr dyy = f(x, y + 1, z, n) - 2 * f(x, y, z, n) + f(x, y - 1, z, n);
    Expr dss = f(x, y, z + 1, n) - 2 * f(x, y, z, n) + f(x, y, z - 1, n);

    Expr dxy = (f(x + 1, y + 1, z, n) - f(x - 1, y - 1, z, n) -  //
                f(x + 1, y - 1, z, n) + f(x - 1, y - 1, z, n)) /
               4;

    Expr dxs = (f(x + 1, y, z + 1, n) - f(x - 1, y, z + 1, n) -  //
                f(x + 1, y, z - 1, n) + f(x - 1, y, z - 1, n)) /
               4;

    Expr dys = (f(x, y + 1, z + 1, n) - f(x, y - 1, z + 1, n) -  //
                f(x, y + 1, z - 1, n) + f(x, y - 1, z - 1, n)) /
               4;

    auto h = Matrix<3, 3>{};
    h(0, 0) = dxx; h(0, 1) = dxy; h(0, 2) = dxs;
    h(1, 0) = dxy; h(1, 1) = dyy; h(1, 2) = dys;
    h(2, 0) = dxs; h(2, 1) = dys; h(2, 2) = dss;

    return h;
  }

  template <typename Input>
  auto scale_space_gradient(const Input& in0,       //
                            const Input& in1,       //
                            const Input& in2,       //
                            const Halide::Expr& x,  //
                            const Halide::Expr& y)  //
      -> Vector<3>
  {
    auto g = Vector<3>{};
    g(0) = (in1(x + 1, y) - in1(x - 1, y)) / 2;
    g(1) = (in1(x, y + 1) - in1(x, y - 1)) / 2;
    g(2) = (in2(x, y) - in0(x, y)) / 2;
    return g;
  }

  template <typename Input>
  auto scale_space_hessian(const Input& in0,       //
                           const Input& in1,       //
                           const Input& in2,       //
                           const Halide::Expr& x,  //
                           const Halide::Expr& y)  //
      -> Matrix<3, 3>
  {
    using Halide::Expr;

    const Expr dxx = in1(x + 1, y) - 2 * in1(x, y) + in1(x - 1, y);
    const Expr dyy = in1(x, y + 1) - 2 * in1(x, y) + in1(x, y - 1);
    const Expr dss = in2(x, y) - 2 * in1(x, y) + in0(x, y);

    const Expr dxy = (in1(x + 1, y + 1) - in1(x - 1, y - 1) -  //
                      in1(x + 1, y - 1) + in1(x - 1, y - 1)) /
                     4;

    const Expr dxs = (in2(x + 1, y) - in2(x - 1, y) -  //
                      in0(x + 1, y) + in0(x - 1, y)) /
                     4;

    const Expr dys = (in2(x, y + 1) - in2(x, y - 1) -  //
                      in0(x, y + 1) + in0(x, y - 1)) /
                     4;

    auto h = Matrix<3, 3>{};
    h(0, 0) = dxx; h(0, 1) = dxy; h(0, 2) = dxs;
    h(1, 0) = dxy; h(1, 1) = dyy; h(1, 2) = dys;
    h(2, 0) = dxs; h(2, 1) = dys; h(2, 2) = dss;

    return h;
  }
  //! @}


  //! @brief Bad API.
  //! @{
  template <typename Input>
  auto gradient(const Input& in,                               //
                const Halide::Expr& x, const Halide::Expr& y,  //
                const Halide::Var& c, const Halide::Var& n)    //
      -> Vector<2>
  {
    auto g = Vector<2>{};
    g(0) = (in(x + 1, y, c, n) - in(x - 1, y, c, n)) / 2;
    g(1) = (in(x, y + 1, c, n) - in(x, y - 1, c, n)) / 2;
    return g;
  }

  template <typename Input>
  auto hessian(const Input& in,                               //
               const Halide::Expr& x, const Halide::Expr& y,  //
               const Halide::Expr& c, const Halide::Expr& n)  //
      -> Matrix<2, 2>
  {
    auto dxx = in(x + 1, y, c, n) + in(x - 1, y, c, n) - 2 * in(x, y, c, n);
    auto dyy = in(x, y + 1, c, n) + in(x, y - 1, c, n) - 2 * in(x, y, c, n);

    auto dxy = (in(x + 1, y + 1, c, n) - in(x - 1, y - 1, c, n) -  //
                in(x + 1, y - 1, c, n) + in(x - 1, y - 1, c, n)) /
               4;

    auto h = Matrix2{};
    h(0, 0) = dxx; h(0, 1) = dxy;
    h(1, 0) = dxy; h(1, 1) = dyy;

    return h;
  }

  template <typename Input>
  auto laplacian(const Input& in,                               //
                 const Halide::Expr& x, const Halide::Expr& y,  //
                 const Halide::Expr& c, const Halide::Expr& n)  //
      -> Halide::Expr
  {
    return in(x + 1, y, c, n) + in(x - 1, y, c, n) - 2 * in(x, y, c, n);
  }

  template <typename Input>
  auto scale_space_gradient(const Input& in0,                              //
                            const Input& in1,                              //
                            const Input& in2,                              //
                            const Halide::Expr& x, const Halide::Expr& y,  //
                            const Halide::Expr& c, const Halide::Expr& n)  //
      -> Vector<3>
  {
    auto g = Vector<3>{};
    g(0) = (in1(x + 1, y, c, n) - in1(x - 1, y, c, n)) / 2;
    g(1) = (in1(x, y + 1, c, n) - in1(x, y - 1, c, n)) / 2;
    g(2) = (in2(x, y, c, n) - in0(x, y, c, n)) / 2;
    return g;
  }

  template <typename Input>
  auto scale_space_hessian(const Input& in0,              //
                           const Input& in1,              //
                           const Input& in2,              //
                           const Halide::Expr& x, const Halide::Expr& y,  //
                           const Halide::Expr& c, const Halide::Expr& n)  //
      -> Matrix<3, 3>
  {
    using Halide::Expr;

    Expr dxx = in1(x + 1, y, c, n) - 2 * in1(x, y, c, n) + in1(x - 1, y, c, n);
    Expr dyy = in1(x, y + 1, c, n) - 2 * in1(x, y, c, n) + in1(x, y - 1, c, n);
    Expr dss = in2(x, y, c, n) - 2 * in1(x, y, c, n) + in0(x, y, c, n);

    Expr dxy = (in1(x + 1, y + 1, c, n) - in1(x - 1, y - 1, c, n) -  //
                in1(x + 1, y - 1, c, n) + in1(x - 1, y - 1, c, n)) /
               4;

    Expr dxs = (in2(x + 1, y, c, n) - in2(x - 1, y, c, n) -  //
                in0(x + 1, y, c, n) + in0(x - 1, y, c, n)) /
               4;

    Expr dys = (in2(x, y + 1, c, n) - in2(x, y - 1, c, n) -  //
                in0(x, y + 1, c, n) + in0(x, y - 1, c, n)) /
               4;

    auto h = Matrix<3, 3>{};
    h(0, 0) = dxx; h(0, 1) = dxy; h(0, 2) = dxs;
    h(1, 0) = dxy; h(1, 1) = dyy; h(1, 2) = dys;
    h(2, 0) = dxs; h(2, 1) = dys; h(2, 2) = dss;

    return h;
  }
  //! @}

}  // namespace DO::Shakti::HalideBackend
