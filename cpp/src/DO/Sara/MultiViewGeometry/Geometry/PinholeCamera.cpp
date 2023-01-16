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

#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>


namespace DO::Sara {

  auto cheirality_predicate(const MatrixXd& X)
      -> Eigen::Array<bool, 1, Eigen::Dynamic>
  {
    if (X.rows() == 3)
      return X.row(2).array() > 0;

    if (X.rows() == 4)
      return X.colwise().hnormalized().row(2).array() > 0;

    throw std::runtime_error{"Error: invalid 3D coordinates!"};
  }

  auto relative_cheirality_predicate(const MatrixXd& X, const Matrix34d& P)
      -> Eigen::Array<bool, 1, Eigen::Dynamic>
  {
    if (X.rows() != 3 && X.rows() != 4)
      throw std::runtime_error{"Error: invalid 3D coordinates!"};

#ifdef OLD
    // The center of the camera has coordinates [0, 0, 0] in the camera frame.
    // Its coordinates in the reference frame satisfies the relation R C + t =
    // 0.
    const auto R = P.block<3, 3>(0, 0);
    const auto t = P.col(3);
    const Eigen::Vector3d C = -R.transpose() * t;

    // The z-axis of the second camera has coordinates [0, 0, 1] in the camera
    // frame Its coordinates in the reference frame satisfies the relation R z =
    // [0, 0, 1]^T and is calculated as
    const Eigen::RowVector3d e3 = R.row(2);

    // The ray (X - C) and the z-axis of the second camera must form a positive
    // angle. In terms of calculation, its dot product must be positive.
    if (X.rows() == 3)
      return (e3 * (X.colwise() - C)).array() > 0;

    const auto X_euclidean = X.topRows(3);
    return (e3 * (X_euclidean.colwise() - C)).array() > 0;
#else
    const auto R = P.block<3, 3>(0, 0);
    const auto t = P.col(3);

    auto X2 = Eigen::MatrixXd{};
    if (X.rows() == 3)
      X2 = (R * X).colwise() + t;
    else
      X2 = (R * X.colwise().hnormalized()).colwise() + t;

    return X2.row(2).array() > 0;
#endif
  }

  auto relative_motion_cheirality_predicate(const Eigen::MatrixXd& X,
                                            const Matrix34d& P)
      -> Eigen::Array<bool, 1, Eigen::Dynamic>
  {
    // The 3D points must be in front of both cameras.
    const auto in_front_of_camera_1 = cheirality_predicate(X);
    const auto in_front_of_camera_2 = relative_cheirality_predicate(X, P);

#if 0
    SARA_DEBUG << "z1 > 0 = " << in_front_of_camera_1 << std::endl;
    SARA_DEBUG << "z2 > 0 = " << in_front_of_camera_2 << std::endl;
#endif

    return in_front_of_camera_1 && in_front_of_camera_2;
  }

} /* namespace DO::Sara */
