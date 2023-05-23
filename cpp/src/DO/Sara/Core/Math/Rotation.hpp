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

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  if defined(__has_warning)  // clang
#    if __has_warning("-Wconversion")
#      pragma GCC diagnostic ignored "-Wconversion"
#    endif
#  endif
#endif
#include <Eigen/Geometry>
#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

namespace DO::Sara {

  //! @{
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
  //! @}

  template <typename T>
  inline auto angular_distance(const Eigen::Matrix<T, 3, 3>& R1,
                               const Eigen::Matrix<T, 3, 3>& R2) -> T
  {
    const Eigen::Matrix3f delta = R1 * R2.transpose();
    const auto cosine = T(0.5) * (delta.trace() - T(1));
    return std::acos(cosine);
  }

  template <typename T>
  inline auto calculate_yaw_pitch_roll(const Eigen::Quaternion<T>& q)
      -> Eigen::Vector3<T>
  {
    // roll (x-axis rotation)
    const auto sinr_cosp = 2 * (q.w() * q.x() + q.y() * q.z());
    const auto cosr_cosp = 1 - 2 * (q.x() * q.x() + q.y() * q.y());
    const auto roll = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    const auto sinp = 2 * (q.w() * q.y() - q.z() * q.x());
    const auto pitch =
        std::abs(sinp) >= 1
            ? std::copysign(M_PI / 2, sinp)  // use 90 degrees if out of range
            : std::asin(sinp);

    // yaw (z-axis rotation)
    const auto siny_cosp = 2 * (q.w() * q.z() + q.x() * q.y());
    const auto cosy_cosp = 1 - 2 * (q.y() * q.y() + q.z() * q.z());
    const auto yaw = std::atan2(siny_cosp, cosy_cosp);

    return {yaw, pitch, roll};
  }

}  // namespace DO::Sara
