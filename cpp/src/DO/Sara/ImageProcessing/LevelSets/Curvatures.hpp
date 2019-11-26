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


namespace DO::Sara {

constexpr auto IMAGE_SCHEME_EPS = 1e-6;

//! @brief Evaluate the mean curvature at point p of the isosurface u = 0.
//!
//! The mean curvature formula is
//!   (∇u Hu ∇u.T - 2 * |∇u|^2 trace(Hu)) / (2 * |∇u|^3).
template <typename T, int N>
T mean_curvature(const Image<T, N>& u, const Matrix<int, N, 1>& p)
{
  const Matrix<T, 3, 1> du = gradient(u, p);
  const Matrix<T, 3, 3> d2u = hessian(u, p);
  const auto du_norm_2 = du.squaredNorm();
  const auto du_norm_3 = du_norm_2 * du.norm();

  if (du_norm_2 < T(IMAGE_SCHEME_EPS))
    return 0;

  return (du.transpose() * d2u * du - du_norm_2 * d2u.trace()) /
         (2 * du_norm_3);
}


//! Evaluate the mean curvature motion at point p of the isosurface u = 0.
//! The mean curvature motion is:
//!   (∇u Hu ∇u.T - 2 * |∇u|^2 trace(Hu)) / (2 * |∇u|^2).
template <typename T, int N>
T mean_curvature_motion(const Image<T, N>& u, const Matrix<int, N, 1>& p)
{
  const Matrix<T, 3, 1> du = gradient(u, p);
  const Matrix<T, 3, 3> d2u = hessian(u, p);
  const auto du_norm_2 = du.squaredNorm();

  if (du_norm_2 < T(IMAGE_SCHEME_EPS))
    return 0;

  return (du.transpose() * d2u * du - du_norm_2 * d2u.trace()) /
         (2 * du_norm_2);
}


template <typename T>
T gaussian_curvature(const Image<T, 3>& u, Vector3i& p)
{
  const Matrix<T, 3, 1> du = gradient(u, p);
  const Matrix<T, 3, 3> d2u = hessian(u, p);

  const auto& ux = du(0);
  const auto& uy = du(1);
  const auto& uz = du(2);

  const auto& uxx = d2u(0, 0); 
  const auto& uyy = d2u(1, 1); 
  const auto& uzz = d2u(2, 2); 
  const auto& uxy = d2u(0, 1); 
  const auto& uyz = d2u(1, 2); 
  const auto& uzx = d2u(2, 0); 

  const auto ux2 = ux * ux;
  const auto uy2 = uy * uy;
  const auto uz2 = uz * uz;
  const auto grad = du.squaredNorm();

  if (grad < T(IMAGE_SCHEME_EPS))
    return 0;

  return (ux2 * (uyy * uzz - uyz * uyz) +
          uy2 * (uxx * uzz - uzx * uzx) +
          uz2 * (uxx * uyy - uxy * uxy) +
          2 * (ux * uy * (uzx * uyz - uxy * uzz) +
               uy * uz * (uxy * uzx - uyz * uxx) +
               ux * uz * (uxy * uyz - uzx * uyy))) /
         (grad * grad);
}

}  // namespace DO::Sara
