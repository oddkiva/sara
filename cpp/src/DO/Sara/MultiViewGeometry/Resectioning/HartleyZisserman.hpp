// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Math/RQFactorization.hpp>

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>


namespace DO::Sara {

  template <typename T>
  auto resectioning_hartley_zisserman(const TensorView_<T, 2>& X,
                                      const TensorView_<T, 2>& x)
      -> PinholeCamera
  {
    if (X.rows() != x.rows())
      throw std::runtime_error{"X and x must have the same number of points!"};

    const Eigen::Matrix<T, 1, 4> zero_4 = Eigen::Matrix<T, 1, 4>::Zero();

    auto A = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>{X.rows() * 2, 12};
    for (auto r = 0; r < X.rows(); ++r)
    {
      const auto Xi_T = Eigen::Matrix<T, 1, 4>{X(r, 0), X(r, 1), X(r, 2), 1};
      const auto ui = x(r, 0);
      const auto vi = x(r, 1);

      A.row(2 * r + 0) << Xi_T, zero_4, -ui * Xi_T;
      A.row(2 * r + 1) << zero_4, Xi_T, -vi * Xi_T;
    }

    const auto svd = A.jacobiSvd(Eigen::ComputeFullV);
    const auto P_flat = svd.matrixV().col(11);

    auto P = Eigen::Matrix<T, 3, 4>{};
    P.row(0) = P_flat.template segment<4>(0);
    P.row(1) = P_flat.template segment<4>(4);
    P.row(2) = P_flat.template segment<4>(8);

    const Eigen::Matrix<T, 3, 3> M = P.template block<3, 3>(0, 0);

    auto K = Eigen::Matrix<T, 3, 3>{};
    auto R = Eigen::Matrix<T, 3, 3>{};
    auto t = Eigen::Matrix<T, 3, 1>{};

    // rq_factorization_3x3(M, K, R);
    rq_factorization(M, K, R);

    // Now flip the axes of K.
    const Eigen::Matrix<T, 3, 1> S = K.diagonal().array().sign().matrix();
    K = K * S.asDiagonal();
    R = S.asDiagonal() * R;

    t = K.inverse() * P.col(3);

    // Recover the scale of K.
    const auto scale = K(2, 2);
    K /= scale;

    return {K, R, t};
  }

}  // namespace DO::Sara
