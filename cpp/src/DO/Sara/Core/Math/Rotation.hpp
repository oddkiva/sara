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

#include <Eigen/Geometry>


namespace DO::Sara {

  //! @brief Using autonomous driving convention.
  template <typename T>
  inline auto yaw(T psi) -> Eigen::Matrix<T, 3, 3>
  {
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    return Eigen::AngleAxis<T>{psi, Vec3::UnitZ()}.matrix();
  }

  template <typename T>
  inline auto pitch(T theta) -> Eigen::Matrix<T, 3, 3>
  {
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    return Eigen::AngleAxis<T>{theta, Vec3::UnitY()}.matrix();
  }

  template <typename T>
  inline auto roll(T phi) -> Eigen::Matrix<T, 3, 3>
  {
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    return Eigen::AngleAxis<T>{phi, Vec3::UnitX()}.matrix();
  }

  template <typename T>
  inline auto rotation(T psi, T theta, T phi) -> Eigen::Matrix<T, 3, 3>
  {
    return yaw(psi) * pitch(theta) * roll(phi);
  }

  template <typename T>
  inline auto angular_distance(const Eigen::Matrix<T, 3, 3>& R1,
                               const Eigen::Matrix<T, 3, 3>& R2) -> T
  {
    const Eigen::Matrix3f delta = R1 * R2.transpose();
    const auto cosine = T(0.5) * (delta.trace() - T(1));
    return std::acos(cosine);
  }
}
