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

  //! We should stop using these functions and it won't work on 360 cameras or
  //! ultra-wide fisheye cameras!
  auto cheirality_predicate(const MatrixXd& X)
      -> Eigen::Array<bool, 1, Eigen::Dynamic>
  {
    if (X.rows() == 3)
      return X.row(2).array() > 0;

    if (X.rows() == 4)
      return X.colwise().hnormalized().row(2).array() > 0;

    throw std::runtime_error{"Error: invalid 3D coordinates!"};
  }

  //! We should stop using these functions and it won't work on 360 cameras or
  //! ultra-wide fisheye cameras!
  auto relative_cheirality_predicate(const MatrixXd& X, const Matrix34d& P)
      -> Eigen::Array<bool, 1, Eigen::Dynamic>
  {
    if (X.rows() != 3 && X.rows() != 4)
      throw std::runtime_error{"Error: invalid 3D coordinates!"};

    const auto R = P.block<3, 3>(0, 0);
    const auto t = P.col(3);

    auto X2 = Eigen::MatrixXd{};
    if (X.rows() == 3)
      X2 = (R * X).colwise() + t;
    else
      X2 = (R * X.colwise().hnormalized()).colwise() + t;

    return X2.row(2).array() > 0;
  }

  auto relative_motion_cheirality_predicate(const Eigen::MatrixXd& X,
                                            const Matrix34d& P)
      -> Eigen::Array<bool, 1, Eigen::Dynamic>
  {
    // The 3D points must be in front of both cameras.
    const auto in_front_of_camera_1 = cheirality_predicate(X);
    const auto in_front_of_camera_2 = relative_cheirality_predicate(X, P);

    return in_front_of_camera_1 && in_front_of_camera_2;
  }

} /* namespace DO::Sara */
