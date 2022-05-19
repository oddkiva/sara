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

#include <Eigen/Core>

#include <ceres/ceres.h>
#include <ceres/rotation.h>


namespace DO::Sara {

  struct OmnidirectionalCameraReprojectionError
  {
    static constexpr auto residual_dimension = 2;
    static constexpr auto intrinsic_parameter_count = 10;
    static constexpr auto extrinsic_parameter_count = 6;

    inline OmnidirectionalCameraReprojectionError(double imaged_x,
                                                  double imaged_y,  //
                                                  double scene_x,
                                                  double scene_y)
      : image_point{imaged_x, imaged_y}
      , scene_point{scene_x, scene_y}
    {
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
      const auto scene_coords = Eigen::Matrix<T, 3, 1>{
          static_cast<T>(scene_point.x()),  //
          static_cast<T>(scene_point.y()),  //
          T{}                               //
      };
      auto camera_coords = Eigen::Matrix<T, 3, 1>{};
      ceres::AngleAxisRotatePoint(extrinsics, scene_coords.data(),
                                  camera_coords.data());

      // b) extrinsics[3, 4, 5] are the translation.
      const auto t = Eigen::Map<const Eigen::Matrix<T, 3, 1>>{extrinsics + 3};
      camera_coords += t;


      // Mirror transformation
      //
      // 1. Project on the unit sphere (reflection from the spherical mirror).
      const Vector3 Xs = camera_coords.normalized();
      // 2. Change coordinates.
      const auto& xi = intrinsics[10];
      const Vector3 Xe = Xs + xi * Vector3::UnitZ();
      // 3. Project the reflected ray by the mirror to the normalized plane z
      // = 1.
      const Vector2 m = Xe.hnormalized();

      // Distortion.
      const auto& k1 = intrinsics[5];
      const auto& k2 = intrinsics[6];
      const auto& k3 = intrinsics[7];
      const auto& p1 = intrinsics[8];
      const auto& p2 = intrinsics[9];

      // Radial component (additive).
      const auto r2 = m.squaredNorm();
      const auto r4 = r2 * r2;
      const auto r6 = r4 * r2;
      const Vector2 radial_factor = m * (k1 * r2 + k2 * r4 + k3 * r6);

      // Tangential component (additive).
      static constexpr auto two = static_cast<T>(2);
      const auto tx = two * p1 * m.x() * m.y() + p2 * (r2 + two * p1 * m.x());
      const auto ty = p1 * (r2 + two * p1 * m.y()) + two * p2 * m.x() * m.y();

      // Apply the distortion.
      const Vector2 m_distorted = m + radial_factor + Vector2{tx, ty};

      // Apply the calibration matrix.
      const auto& fx = intrinsics[0];
      const auto& fy = intrinsics[1];
      const auto& s = intrinsics[2];
      const auto& u0 = intrinsics[3];
      const auto& v0 = intrinsics[4];
      // clang-format off
      const auto predicted_x = fx * m_distorted.x() +  s * m_distorted.y() + u0;
      const auto predicted_y =                        fy * m_distorted.y() + v0;
      // clang-format on

      // The error is the difference between the predicted and observed
      // position.
      residuals[0] = predicted_x - static_cast<T>(image_point[0]);
      residuals[1] = predicted_y - static_cast<T>(image_point[1]);

      return true;
    }

    static inline auto create(const double image_x, const double image_y,
                              const double scene_x, const double scene_y)
    {
      return new ceres::AutoDiffCostFunction<
          OmnidirectionalCameraReprojectionError,  //
          residual_dimension, intrinsic_parameter_count,
          extrinsic_parameter_count>(
          new OmnidirectionalCameraReprojectionError(image_x, image_y,  //
                                                     scene_x, scene_y)  //
      );
    }

    Eigen::Vector2d image_point;
    Eigen::Vector2d scene_point;
  };

}  // namespace DO::Sara
