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

#include <DO/Sara/MultiViewGeometry/Estimators/Triangulation.hpp>


namespace DO::Sara {

auto triangulate_linear_eigen(const Matrix34d& P1, const Matrix34d& P2,
                              const Vector3d& u1, const Vector3d& u2)
    -> Vector4d
{
  Matrix<double, 6, 6> M = Matrix<double, 6, 6>::Zero(6, 6);

  for (int i = 0; i < u1.cols(); ++i)
  {
    M.block<3, 4>(0, 0) = P1;
    M.block<3, 4>(3, 0) = P2;
    M.block<3, 1>(0, 4) = -u1;
    M.block<3, 1>(3, 5) = -u2;
  }

  JacobiSVD<MatrixXd> svd(M, Eigen::ComputeFullV);
  const MatrixXd& V = svd.matrixV();
  return V.col(5).head(4) / V(3, 5);
}

} /* namespace DO::Sara */
