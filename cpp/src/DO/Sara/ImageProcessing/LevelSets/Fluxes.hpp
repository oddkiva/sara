// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/ImageProcessing/LevelSets/FiniteDifferences.hpp>


namespace DO { namespace Sara {

  constexpr double flux_delta = 1.;

  //! @brief Evaluate the advection value <v, ∇u> at point p.
  //! - v is the velocity field value evaluated at point p.
  //! - ∇ is the spatial gradient operator.
  template <typename FiniteDifference, typename T, int N>
  inline T advection(const ImageView<T, N>& u, const Matrix<int, N, 1>& p,
                     const Matrix<T, N, 1>& v)
  {
    T delta = 0;
    // See explanation here:
    // https://scicomp.stackexchange.com/questions/27737/advection-equation-with-finite-difference-importance-of-forward-backward-or-ce
    for (int i = 0; i < N; ++i)
      delta -= v[i] * ((v[i] > 0) ? FiniteDifference::backward(u, p, i)
                                  : FiniteDifference::forward(u, p, i));
    return delta;
  }

  //! @brief Evaluate the normal motion value β * |∇u|^2 at point p.
  template <typename FiniteDifference, typename T, int N>
  inline T normal_motion(const ImageView<T, N>& u, const Matrix<int, N, 1>& p,
                         const T beta)
  {
    T delta = 0;

    for (int i = 0; i < N; ++i)
    {
      const T upi = FiniteDifference::forward(u, p, i);
      const T umi = FiniteDifference::backward(u, p, i);

      if (beta > 0)
      {
        if (upi < 0)
          delta += upi * upi;
        if (umi > 0)
          delta += umi * umi;
      }

      else
      {
        if (upi > 0)
          delta += upi * upi;

        if (umi < 0)
          delta += umi * umi;
      }
    }

    return -beta * std::sqrt(delta);
  }

  //! @brief Evaluate ∇u/|∇u| at point p.
  template <typename FiniteDifference, typename T, typename U, int N>
  inline Matrix<T, N, 1> normal(const ImageView<T, N>& u, const Matrix<int, N, 1>& p)
  {
    auto n = Matrix<T, N, 1>::Zero();
    for (auto i = 0; i < N; ++i)
      n(i) = Centered::centered(u, p, i);
    return n / n.norm();
  }

  template <typename FiniteDifference, typename T, typename U, int N>
  T extension(const ImageView<T, N>& u, const ImageView<U, N>& d,
              const Matrix<int, N, 1>& p)
  {
    //! Evaluate the normal ∇u/|∇u| at point p.
    const auto v = normal(u, p);

    //! Rescale v with s(p) = u(p) / sqrt(df^2 + u(p)^2).
    const T u0 = u(p);
    const T s = u0 / sqrt(u0 * u0 + T(flux_delta) * T(flux_delta));
    v *= s;

    // Advection operator <v, ∇d> at point p.
    return advection<FiniteDifference>(d, p, v);
  }

  template <typename FiniteDifference, typename T, int N>
  T reinitialization(const ImageView<T, N>& u, const Matrix<int, N, 1>& p)
  {
    const T u0 = u(p);
    const T s = u0 / std::sqrt(u0 * u0 + T(flux_delta) * T(flux_delta));
    return s + normal_motion<FiniteDifference>(u, p, s);
  }

}  // namespace Sara
}  // namespace DO
