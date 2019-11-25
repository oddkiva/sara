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

#include <DO/Sara/Core/Image.hpp>


namespace DO { namespace Sara {

  constexpr double flux_delta = 1.;

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


  template <typename FiniteDifference, typename T, typename U, int N>
  T extension(const ImageView<T, N>& u, const Image<U, N>& d,
              const Matrix<int, N, 1>& p)
  {
    Matrix<T, N, 1> v;
    normal(u, p, v);
    const T u0 = u(p);
    const T s = u0 / sqrt(u0 * u0 + T(flux_delta) * T(flux_delta));
    v *= s;

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
