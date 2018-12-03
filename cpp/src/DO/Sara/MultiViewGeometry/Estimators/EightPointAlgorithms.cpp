#pragma once

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  void eight_point_fundamental_matrix(const Matrix<double, 3, 8>& x,
                                      const Matrix<double, 3, 8>& y,
                                      Matrix3d& F)
  {
    // 1. solve the linear system from the 8-point correspondences.
    {
      Matrix<double, 8, 9> A;
      for (int i = 0; i < 8; ++i)
      {
        A.row(i) <<                          //
            x(0, i) * y.col(i).transpose(),  //
            x(1, i) * y.col(i).transpose(),  //
            x(2, i) * y.col(i).transpose();

        //std::cout << x.col(i).transpose() << " " << y.col(i).transpose() << std::endl;
      }

      auto svd = Eigen::JacobiSVD<Matrix<double, 8, 9>>{A, Eigen::ComputeFullV};
      const Matrix<double, 8, 1> S = svd.singularValues();
      const Matrix<double, 9, 1> vec_F = svd.matrixV().col(8).normalized();

      F.row(0) = vec_F.segment(0, 3).transpose();
      F.row(1) = vec_F.segment(3, 3).transpose();
      F.row(2) = vec_F.segment(6, 3).transpose();
    }

    // 2. Enforce the rank-2 constraint of the fundamental matrix.
    {
      auto svd = Eigen::JacobiSVD<Matrix3d>{F, Eigen::ComputeFullU |
                                                   Eigen::ComputeFullV};
      Vector3d D = svd.singularValues();
      D(2) = 0;
      F = svd.matrixU() * D.asDiagonal() * svd.matrixV().transpose();
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

    H << h, 1.0;
  }

} /* namespace Sara */
} /* namespace DO */
