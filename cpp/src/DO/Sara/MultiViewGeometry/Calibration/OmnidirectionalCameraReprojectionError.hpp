// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif

#include <Eigen/Core>

#include <ceres/ceres.h>
#include <ceres/rotation.h>


namespace DO::Sara {

  struct OmnidirectionalCameraReprojectionError
  {
    static constexpr auto residual_dimension = 2;
    static constexpr auto intrinsic_parameter_count = 11;
    static constexpr auto extrinsic_parameter_count = 6;

    inline OmnidirectionalCameraReprojectionError(
        const Eigen::Vector2d& image_point,  //
        const Eigen::Vector3d& scene_point)
      : image_point{image_point}
      , scene_point{scene_point}
    {
    }

    template <typename T>
    inline auto
    apply_mirror_transformation(const Eigen::Matrix<T, 3, 1>& Xc,
                                const T& xi) const -> Eigen::Matrix<T, 2, 1>
    {
      using Vector2 = Eigen::Matrix<T, 2, 1>;
      using Vector3 = Eigen::Matrix<T, 3, 1>;

      // Mirror transformation
      //
      // 1. Project on the unit sphere (reflection from the spherical mirror).
      const Vector3 Xs = Xc.normalized();
      const Vector3 Xe = Xs + xi * Vector3::UnitZ();
      // 2. Project the reflected ray by the mirror to the normalized plane z
      // = 1.
      const Vector2 m = Xe.hnormalized();

      return m;
    }

    template <typename T>
    inline auto apply_distortion(const Eigen::Matrix<T, 2, 1>& mu, const T& k1,
                                 const T& k2, const T& k3, const T& p1,
                                 const T& p2) const -> Eigen::Matrix<T, 2, 1>
    {
      using Vector2 = Eigen::Matrix<T, 2, 1>;

      // Radial component (additive).
      const auto r2 = mu.squaredNorm();
      const auto r4 = r2 * r2;
      const auto r6 = r4 * r2;
      const Vector2 radial_factor = mu * (k1 * r2 + k2 * r4 + k3 * r6);

      // Tangential component (additive).
      static constexpr auto two = 2.;
      const auto tx =
          two * p1 * mu.x() * mu.y() + p2 * (r2 + two * p1 * mu.x());
      const auto ty =
          p1 * (r2 + two * p1 * mu.y()) + two * p2 * mu.x() * mu.y();

      // Apply the distortion.
      return mu + radial_factor + Vector2{tx, ty};
    }

    template <typename T>
    inline auto operator()(const T* const intrinsics, const T* const extrinsics,
                           T* residuals) const -> bool
    {
      using Vector2 = Eigen::Matrix<T, 2, 1>;
      using Vector3 = Eigen::Matrix<T, 3, 1>;

      // 1. Apply [R|t] = extrinsics[...]
      //
      // a) extrinsics[0, 1, 2] are the angle-axis rotation.
      const Vector3 scene_coords = scene_point.template cast<T>();
      auto camera_coords = Vector3{};
      ceres::AngleAxisRotatePoint(extrinsics, scene_coords.data(),
                                  camera_coords.data());

      // b) extrinsics[3, 4, 5] are the translation.
      const auto t = Eigen::Map<const Vector3>{extrinsics + 3};
      camera_coords += t;

      // Project the scene point to the spherical mirror by shooting a ray.
      // The ray leaves the scene point, passes through the optical center and
      // hits the spherical mirror.
      const auto& xi = intrinsics[5];
      const auto m = apply_mirror_transformation(camera_coords, xi);

      // The ray is then reflected by the spherical mirror and finally hits the
      // image plane.
      //
      // The spherical mirror or the image plane has an imperfect geometry, so
      // apply the distortion. (?).
      //
      // TODO: re-read the paper just to make sure we understood right.
      const auto& k1 = intrinsics[6];
      const auto& k2 = intrinsics[7];
      const auto& k3 = intrinsics[8];
      const auto& p1 = intrinsics[9];
      const auto& p2 = intrinsics[10];
      const Vector2 m_distorted = apply_distortion(m, k1, k2, k3, p1, p2);

      // Apply the calibration matrix to get the pixel coordinates of the point
      // of impact.
      const auto& fx = intrinsics[0];
      const auto fy_normalized = intrinsics[1];
      const auto fy = fy_normalized * fx;
      const auto& alpha = intrinsics[2];
      const auto shear = fx * alpha;
      const auto& u0 = intrinsics[3];
      const auto& v0 = intrinsics[4];
      // clang-format off
      const auto predicted_x = fx * m_distorted.x() + shear * m_distorted.y() + u0;
      const auto predicted_y =                           fy * m_distorted.y() + v0;
      // clang-format on

      // The error is the difference between the predicted and observed
      // position.
      residuals[0] = predicted_x - static_cast<T>(image_point[0]);
      residuals[1] = predicted_y - static_cast<T>(image_point[1]);

      return true;
    }

    static inline auto create(const Eigen::Vector2d& image_point,
                              const Eigen::Vector3d& scene_point)
    {
      return new ceres::AutoDiffCostFunction<
          OmnidirectionalCameraReprojectionError,  //
          residual_dimension, intrinsic_parameter_count,
          extrinsic_parameter_count>(
          new OmnidirectionalCameraReprojectionError(image_point, scene_point));
    }

    Eigen::Vector2d image_point;
    Eigen::Vector3d scene_point;
  };

}  // namespace DO::Sara
