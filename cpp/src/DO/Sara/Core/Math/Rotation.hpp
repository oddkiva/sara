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
  inline auto yaw(T psi) -> Matrix<T, 3, 3>
  {
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    return Eigen::AngleAxis<T>{psi, Vec3::UnitZ()}.matrix();
  }

  template <typename T>
  inline auto pitch(T theta) -> Matrix<T, 3, 3>
  {
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    return Eigen::AngleAxis<T>{theta, Vec3::UnitY()}.matrix();
  }

  template <typename T>
  inline auto roll(T phi) -> Matrix<T, 3, 3>
  {
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    return Eigen::AngleAxis<T>{phi, Vec3::UnitX()}.matrix();
  }

  template <typename T>
  inline auto rotation(T psi, T theta, T phi) -> Matrix<T, 3, 3>
  {
    return yaw(psi) * pitch(theta) * roll(phi);
  }

}
