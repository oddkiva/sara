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


namespace DO::Sara {

  //! @brief Evaluate the normal ∇u/|∇u| at point x.
  template <typename T, int N>
  inline auto normal(const ImageView<T, N>& u,    //
                     const Matrix<int, N, 1>& x,  //
                     T eps = 1e-6)                //
      -> Matrix<T, N, 1>
  {
    auto n = Matrix<T, N, 1>{};
    for (auto i = 0; i < N; ++i)
      n(i) = Centered::centered(u, x, i);

    auto norm_n = n.norm();
    if (norm_n < eps)
      norm_n = eps;

    n /= norm_n;

    return n;
  }

  //! @brief Evaluate the advection value <v, ∇u> at point x.
  //! - v is the velocity field value evaluated at point x.
  //! - ∇ is the spatial gradient operator.
  template <typename FiniteDifference, typename T, int N>
  inline auto advection(const ImageView<T, N>& u,    //
                        const Matrix<int, N, 1>& x,  //
                        const Matrix<T, N, 1>& v)    //
      -> T
  {
    auto delta = T(0);
    // See explanation about the sign here:
    //
    // https://scicomp.stackexchange.com/questions/27737/advection-equation-with-finite-difference-importance-of-forward-backward-or-ce
    for (int i = 0; i < N; ++i)
      delta -= v[i] * ((v[i] > 0) ? FiniteDifference::backward(u, x, i)
                                  : FiniteDifference::forward(u, x, i));
    return delta;
  }

  //! @brief Evaluate the normal motion value β * |∇u| at point x.
  template <typename FiniteDifference, typename T, int N>
  inline auto normal_motion(const ImageView<T, N>& u,    //
                            const Matrix<int, N, 1>& x,  //
                            const T beta)                //
      -> T
  {
    auto delta = T(0);

    for (int i = 0; i < N; ++i)
    {
      const auto upi = FiniteDifference::forward(u, x, i);
      const auto umi = FiniteDifference::backward(u, x, i);

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

  //! @brief Evaluate the extension operator β * |∇u| at point x.
  template <typename FiniteDifference, typename T, typename U, int N>
  inline auto extension(const ImageView<T, N>& u,    //
                        const ImageView<U, N>& d,    //
                        const Matrix<int, N, 1>& x,  //
                        T flux_delta = 1)            //
      -> T
  {
    //! Evaluate the velocity direction as the normal ∇u/|∇u| at point x.
    const auto v = normal(u, x);

    //! Evaluate the magnitude of the velocity as:
    //!   u(x) / sqrt(df^2 + u(x)^2).
    const auto& u0 = u(x);
    const auto s = u0 / sqrt(u0 * u0 + flux_delta * flux_delta);
    v *= s;

    // Advection operator <v, ∇d> at point x.
    return advection<FiniteDifference>(d, x, v);
  }

  template <typename FiniteDifference, typename T, int N>
  inline auto reinitialization(const ImageView<T, N>& u,    //
                               const Matrix<int, N, 1>& x,  //
                               T flux_delta = 1)            //
      -> T
  {
    const auto& u0 = u(x);
    const auto s = u0 / std::sqrt(u0 * u0 + flux_delta * flux_delta);
    return s + normal_motion<FiniteDifference>(u, x, s);
  }

}  // namespace DO::Sara
