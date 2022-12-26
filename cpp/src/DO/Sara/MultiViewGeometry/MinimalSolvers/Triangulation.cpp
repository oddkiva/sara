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
#include <DO/Sara/MultiViewGeometry/Estimators/Triangulation.hpp>


namespace DO::Sara {

  auto triangulate_single_point_linear_eigen(const Matrix34d& P1,
                                             const Matrix34d& P2,
                                             const Vector3d& ray1,
                                             const Vector3d& ray2)
      -> std::tuple<Vector4d, double, double>
  {
    Matrix<double, 6, 6> M = Matrix<double, 6, 6>::Zero(6, 6);

    for (int i = 0; i < ray1.cols(); ++i)
    {
      M.block<3, 4>(0, 0) = P1;
      M.block<3, 4>(3, 0) = P2;
      M.block<3, 1>(0, 4) = -ray1;
      M.block<3, 1>(3, 5) = -ray2;
    }

    JacobiSVD<MatrixXd> svd(M, Eigen::ComputeFullV);
    const MatrixXd& V = svd.matrixV();

    // V.col(5) is by definition [ X, s1, s2 ].
    // where:
    // - P1 * X = s1 * ray1
    // - P2 * X = s2 * ray2
    //
    // Here we also return the scales s1 and s2, because they can tell whether
    // the cheirality constraints are satisfied.
    //
    // In words, the cheirality constraints are:
    // - s1 > 0
    // - s2 > 0

    const auto V_rescaled = V.col(5) / V(3, 5);

    // The 3D scene points.
    const Eigen::Vector4d X = V_rescaled.head(4);
    // The scales for each ray.
    const auto s1 = V_rescaled(4);
    const auto s2 = V_rescaled(5);

    return std::make_tuple(X, s1, s2);
  }

  auto triangulate_linear_eigen(const Matrix34d& P1, const Matrix34d& P2,
                                const MatrixXd& u1, const MatrixXd& u2)
      -> std::tuple<MatrixXd, VectorXd, VectorXd>
  {
    auto X = MatrixXd{4, u1.cols()};
    auto s1 = VectorXd{u1.cols()};
    auto s2 = VectorXd{u1.cols()};
    for (int i = 0; i < u1.cols(); ++i)
    {
      const auto [Xi, s1i, s2i] = triangulate_single_point_linear_eigen(
          P1, P2, Vector3d{u1.col(i)}, Vector3d{u2.col(i)});

      X.col(i) = Xi;
      s1(i) = s1i;
      s2(i) = s2i;
    }
    return std::make_tuple(X, s1, s2);
  }

} /* namespace DO::Sara */
