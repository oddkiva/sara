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

#pragma once

#include <Eigen/Core>


namespace DO::Kalpana {

  template <typename T>
  inline auto frustum(const T l, const T r, const T b, const T t, const T n,
                      const T f) -> Eigen::Matrix4<T>
  {
    auto proj = Eigen::Matrix4<T>{};

    // clang-format off
    proj <<
      2*n/(r-l),         0,  (r+l)/(r-l),            0,
              0, 2*n/(t-b),  (t+b)/(t-b),            0,
              0,         0, -(f+n)/(f-n), -2*f*n/(f-n),
              0,         0,           -1,            0;
    // clang-format on

    return proj;
  }

  template <typename T>
  inline auto perspective(const T fov_degrees, const T aspect, const T z_near,
                          const T z_far) -> Eigen::Matrix4<T>
  {
    static constexpr auto to_radians = static_cast<T>(M_PI / 360.);
    const auto t = z_near * std::tan(fov_degrees * to_radians);
    const auto b = -t;
    const auto l = aspect * b;
    const auto r = aspect * t;
    return frustum(l, r, b, t, z_near, z_far);
  }

  template <typename T>
  inline auto orthographic(const T l, const T r, const T b, const T t,
                           const T n, const T f) -> Eigen::Matrix4<T>
  {
    auto proj = Eigen::Matrix4<T>{};
    // clang-format off
    proj <<
      2/(r-l),       0,       0, -(r+l)/(r-l),
            0, 2/(t-b),       0, -(t+b)/(t-b),
            0,       0,-2/(f-n), -(f+n)/(f-n),
            0,       0,       0,            1;
    // clang-format on
    return proj;
  }

  template <typename T>
  inline auto look_at(const Eigen::Vector3<T>& eye,
                      const Eigen::Vector3<T>& center,
                      const Eigen::Vector3<T>& up) -> Eigen::Matrix4<T>
  {
    const Eigen::Vector3<T> f = (center - eye).normalized();
    Eigen::Vector3<T> u = up.normalized();
    const Eigen::Vector3<T> s = f.cross(u).normalized();
    u = s.cross(f);

    auto res = Eigen::Matrix4<T>{};
    // clang-format off
    res <<
       s.x(),  s.y(),  s.z(), -s.dot(eye),
       u.x(),  u.y(),  u.z(), -u.dot(eye),
      -f.x(), -f.y(), -f.z(),  f.dot(eye),
           0,      0,      0,           1;
    // clang-format on

    return res;
  }

}  // namespace DO::Kalpana
