// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018-2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>


namespace DO::Sara {

struct PinholeCamera
{
  operator Matrix34d() const
  {
    Matrix34d Rt;
    Rt.topLeftCorner(3, 3) = R;
    Rt.col(3) = t;
    return K * Rt;
  }

  Eigen::Matrix3d K;
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
};

inline auto normalized_camera(const Eigen::Matrix3d& R,
                              const Eigen::Vector3d& t)
{
  return PinholeCamera{Eigen::Matrix3d::Identity(), R, t};
}

inline auto normalized_camera(const Motion& m)
{
  return PinholeCamera{Eigen::Matrix3d::Identity(), m.R, m.t};
}


DO_SARA_EXPORT
auto cheirality_predicate(const MatrixXd& X, const Matrix34d& P) -> bool;

} /* namespace DO::Sara */
