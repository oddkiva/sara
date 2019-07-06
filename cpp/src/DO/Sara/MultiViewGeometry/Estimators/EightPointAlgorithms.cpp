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


namespace DO { namespace Sara {

  void eight_point_fundamental_matrix(const Matrix<double, 3, 8>& p_left,
                                      const Matrix<double, 3, 8>& p_right,
                                      Matrix3d& F)
  {
    // 1. solve the linear system from the 8-point correspondences.
    {
      Matrix<double, 8, 9> A;
      for (int i = 0; i < 8; ++i)
      {
        A.row(i) <<                          //
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
      F = F.normalized();
    }
  }

  void four_point_homography(const Matrix<double, 3, 4>& x,
                             const Matrix<double, 3, 4>& y,  //
                             Matrix3d& H)
  {
    const auto zero = RowVector3d::Zero();

    auto M = Matrix<double, 8, 8>{};
    for (int i = 0; i < 4; ++i)
    {
      RowVector3d u_i = x.col(i).transpose();
      RowVector3d v_i = y.col(i).transpose();
      M.row(2* i + 0) <<  u_i, zero, - u_i * v_i.x();
      M.row(2* i + 1) << zero,  u_i, - u_i * v_i.y();
    }

    const Matrix<double, 2, 4> y_euclidean = y.topRows<2>();
    const Map<const Matrix<double, 8, 1>> b{y_euclidean.data()};

    const Matrix<double, 8, 1> h = M.colPivHouseholderQr().solve(b);

    H.row(0) = h.segment(0, 3).transpose();
    H.row(1) = h.segment(3, 3).transpose();
    H.row(2) << h.segment(6, 2).transpose(), 1;
    SARA_CHECK(H);
  }

} /* namespace Sara */
} /* namespace DO */
