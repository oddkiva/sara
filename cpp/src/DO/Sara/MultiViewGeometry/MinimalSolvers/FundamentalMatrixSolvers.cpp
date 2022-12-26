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

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/EigenExtension.hpp>

#include <DO/Sara/MultiViewGeometry/MinimalSolvers/FundamentalMatrixSolvers.hpp>


namespace DO::Sara {

auto EightPointAlgorithm::
operator()(const EightPointAlgorithm::matrix_view_type& p_left,
           const EightPointAlgorithm::matrix_view_type& p_right) const
    -> std::array<EightPointAlgorithm::model_type, 1>
{
  auto F = Matrix3d{};

  // 1. solve the linear system from the 8-point correspondences.
  {
    Matrix<double, 8, 9> A;
    for (int i = 0; i < 8; ++i)
    {
      A.row(i) <<                                     //
          p_right(0, i) * p_left.col(i).transpose(),  //
          p_right(1, i) * p_left.col(i).transpose(),  //
          p_right(2, i) * p_left.col(i).transpose();
    }

    auto svd = Eigen::BDCSVD<Matrix<double, 8, 9>>{A, Eigen::ComputeFullV};
    const Matrix<double, 9, 1> vec_F = svd.matrixV().col(8).normalized();

    F.row(0) = vec_F.segment(0, 3).transpose();
    F.row(1) = vec_F.segment(3, 3).transpose();
    F.row(2) = vec_F.segment(6, 3).transpose();
  }

  // 2. Enforce the rank-2 constraint of the fundamental matrix.
  {
    auto svd =
        Eigen::BDCSVD<Matrix3d>{F, Eigen::ComputeFullU | Eigen::ComputeFullV};
    Vector3d D = svd.singularValues();
    D(2) = 0;
    F = svd.matrixU() * D.asDiagonal() * svd.matrixV().transpose();
    F = F.matrix().normalized();
  }

  return {F};
}

} /* namespace DO::Sara */
