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

  struct PinholeCameraReprojectionError
  {
    static constexpr auto residual_dimension = 2;
    static constexpr auto intrinsic_parameter_count = 5;
    static constexpr auto extrinsic_parameter_count = 6;

    inline PinholeCameraReprojectionError(const Eigen::Vector2d& image_pt,  //
                                          const Eigen::Vector3d& scene_pt)
      : image_point{image_pt}
      , scene_point{scene_pt}
    {
    }

    template <typename T>
    inline auto operator()(const T* const intrinsics, const T* const extrinsics,
                           T* residuals) const
        -> bool
    {
      using Vector3 = Eigen::Matrix<T, 3, 1>;
      const Vector3 scene_coords = scene_point.template cast<T>();

      // 1. Apply [R|t] = extrinsics[...]
      //
      // a) extrinsics[0, 1, 2] are the angle-axis rotation.
      auto camera_coords = Eigen::Matrix<T, 3, 1>{};
      ceres::AngleAxisRotatePoint(extrinsics, scene_coords.data(),
                                  camera_coords.data());

      // b) extrinsics[3, 4, 5] are the translation.
      const auto t = Eigen::Map<const Eigen::Matrix<T, 3, 1>>{extrinsics + 3};
      camera_coords += t;

      // 2. Calculate the normalized camera coordinates.
      const auto xp = camera_coords[0] / camera_coords[2];
      const auto yp = camera_coords[1] / camera_coords[2];

      // 3. Apply the calibration matrix.
      const auto& fx = intrinsics[0];
      const auto& fy_normalized = intrinsics[1];
      const auto& fy = fy_normalized * fx;
      const auto& shear_normalized = intrinsics[2];
      const auto& s = shear_normalized * fx;
      const auto& u0 = intrinsics[3];
      const auto& v0 = intrinsics[4];
      // clang-format off
      const auto predicted_x = fx * xp +  s * yp + u0;
      const auto predicted_y =           fy * yp + v0;
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
      return new ceres::AutoDiffCostFunction<PinholeCameraReprojectionError,  //
                                             residual_dimension,              //
                                             intrinsic_parameter_count,
                                             extrinsic_parameter_count>(
          new PinholeCameraReprojectionError(image_point, scene_point)  //
      );
    }

    Eigen::Vector2d image_point;
    Eigen::Vector3d scene_point;
  };

}  // namespace DO::Sara
