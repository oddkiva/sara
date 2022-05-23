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

  template <typename T>
  struct OmnidirectionalCameraParameterVector
  {
    static constexpr auto size = 10;
    enum Index
    {
      FX = 0,
      FY = 1,
      NORMALIZED_SHEAR = 2,
      U0 = 3,
      V0 = 4,
      K0 = 5,
      K1 = 6,
      P0 = 7,
      P1 = 8,
      XI = 9
    };

    const T* data = nullptr;

    inline auto f(const int i) const -> const T&
    {
      return data[FX + i];
    }

    inline auto normalized_shear() const -> const T&
    {
      return data[NORMALIZED_SHEAR];
    }

    inline auto shear() const -> T
    {
      return data[NORMALIZED_SHEAR] * data[FX];
    }

    inline auto u0() const -> const T&
    {
      return data[U0];
    }

    inline auto v0() const -> const T&
    {
      return data[U0];
    }

    inline auto k(const int i) const -> const T&
    {
      return data[K0 + i];
    }

    inline auto p(const int i) const -> const T&
    {
      return data[P0 + i];
    }

    inline auto xi() const -> const T&
    {
      return data[XI];
    }
  };

  struct OmnidirectionalCameraReprojectionError
  {
    static constexpr auto residual_dimension = 2;
    static constexpr auto intrinsic_parameter_count =
        OmnidirectionalCameraParameterVector<double>::size;
    static constexpr auto extrinsic_parameter_count = 6;

    inline OmnidirectionalCameraReprojectionError(
        const Eigen::Vector2d& image_point,  //
        const Eigen::Vector2d& scene_point)
      : image_point{image_point}
      , scene_point{scene_point}
    {
    }

    template <typename T>
    inline auto apply_mirror_transformation(const Eigen::Matrix<T, 3, 1>& Xc,
                                            const T& xi) const
        -> Eigen::Matrix<T, 2, 1>
    {
      using Vector2 = Eigen::Matrix<T, 2, 1>;
      using Vector3 = Eigen::Matrix<T, 3, 1>;

      // Mirror transformation
      //
      // 1. Project on the unit sphere (reflection from the spherical mirror).
      const Vector3 Xs = Xc.normalized();
      const Vector3 Xe = Xs + xi * Vector3::UnitZ();
      // 3. Project the reflected ray by the mirror to the normalized plane z
      // = 1.
      const Vector2 m = Xe.hnormalized();

      return m;
    }

    template <typename T>
    inline auto apply_distortion(const Eigen::Matrix<T, 2, 1>& mu, const T& k1,
                                 const T& k2, const T& p1, const T& p2) const
        -> Eigen::Matrix<T, 2, 1>
    {
      using Vector2 = Eigen::Matrix<T, 2, 1>;

      // Radial component (additive).
      const auto r2 = mu.squaredNorm();
      const auto r4 = r2 * r2;
      const Vector2 radial_factor = mu * (k1 * r2 + k2 * r4);

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

      const auto& xi = intrinsics[9];
      const auto m = apply_mirror_transformation(camera_coords, xi);

      // Distortion.
      const auto& k1 = intrinsics[5];
      const auto& k2 = intrinsics[6];
      const auto& p1 = intrinsics[7];
      const auto& p2 = intrinsics[8];
      const Vector2 m_distorted = apply_distortion(m, k1, k2, p1, p2);

      // Apply the calibration matrix.
      const auto& fx = intrinsics[0];
      const auto& fy = intrinsics[1];
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
                              const Eigen::Vector2d& scene_point)
    {
      return new ceres::AutoDiffCostFunction<
          OmnidirectionalCameraReprojectionError,  //
          residual_dimension, intrinsic_parameter_count,
          extrinsic_parameter_count>(
          new OmnidirectionalCameraReprojectionError(image_point, scene_point));
    }

    Eigen::Vector2d image_point;
    Eigen::Vector2d scene_point;
  };

  struct DistortionParamRegularizer
  {
    static constexpr auto residual_dimension = 4;
    static constexpr auto intrinsic_parameter_count =
        OmnidirectionalCameraParameterVector<double>::size;

    inline DistortionParamRegularizer(double scale = 1.0)
      : scale{scale}
    {
    }

    template <typename T>
    inline auto operator()(const T* const intrinsics, T* residuals) const
        -> bool
    {
      const auto& k1 = intrinsics[5];
      const auto& k2 = intrinsics[6];
      const auto& p1 = intrinsics[7];
      const auto& p2 = intrinsics[8];

      residuals[0] = scale * k1;
      residuals[1] = scale * k2;
      residuals[0] = scale * p1;
      residuals[1] = scale * p2;

      return true;
    }

    static inline auto create(double scale)
    {
      return new ceres::AutoDiffCostFunction<DistortionParamRegularizer,  //
                                             residual_dimension,
                                             intrinsic_parameter_count>(
          new DistortionParamRegularizer{scale});
    }

    double scale;
  };
}  // namespace DO::Sara
