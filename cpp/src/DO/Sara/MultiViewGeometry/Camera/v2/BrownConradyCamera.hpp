// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2023-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/MultiViewGeometry/Camera/v2/CameraIntrinsicBase.hpp>

#include <Eigen/Geometry>


namespace DO::Sara::v2 {

  template <ArrayConcept Array>
  struct BrownConradyDistortionModelBase : CameraIntrinsicBase<Array>
  {
    using base_type = CameraIntrinsicBase<Array>;
    using base_type::data;
    using T = typename Array::value_type;

    // clang-format off
    enum Index
    {
      FX = 0,
      FY = 1,
      S  = 2,
      U0 = 3,
      V0 = 4,
      K0 = 5,
      K1 = 6,
      K2 = 7,
      P0 = 8,
      P1 = 9,
    };
    // clang-format on

    auto fx() -> T&
    {
      return data[FX];
    }

    auto fy() -> T&
    {
      return data[FY];
    }

    auto shear() -> T&
    {
      return data[S];
    }

    auto u0() -> T&
    {
      return data[U0];
    }

    auto v0() -> T&
    {
      return data[V0];
    }

    auto k() -> VectorView<T, 3>
    {
      return VectorView<T, 3>{data.data() + K0};
    }

    auto p() -> VectorView<T, 2>
    {
      return VectorView<T, 2>{data.data() + P0};
    }

    auto k(const int i) -> T&
    {
      return data[K0 + i];
    };

    auto p(const int i) -> T&
    {
      return data[P0 + i];
    };

    auto fx() const -> T
    {
      return data[FX];
    }

    auto fy() const -> T
    {
      return data[FY];
    }

    auto shear() const -> T
    {
      return data[S];
    }

    auto u0() const -> T
    {
      return data[U0];
    }

    auto v0() const -> T
    {
      return data[V0];
    }

    auto k() const -> ConstVectorView<T, 3>
    {
      return ConstVectorView<T, 3>{data.data() + K0};
    }

    auto p() const -> ConstVectorView<T, 2>
    {
      return ConstVectorView<T, 2>{data.data() + P0};
    }

    auto k(const int i) const -> T
    {
      return data[K0 + i];
    };

    auto p(const int i) const -> T
    {
      return data[P0 + i];
    };

    auto to_pixel_coordinates(const Eigen::Vector2<T>& xn) const
        -> Eigen::Vector2<T>
    {
      return {
          fx() * xn.x() + shear() * xn.y() + u0(),  //
          /*              */ fy() * xn.y() + v0()   //
      };
    }

    auto to_normalized_camera_coordinates(const Eigen::Vector2<T>& x) const
        -> Eigen::Vector2<T>
    {
      auto xn = Eigen::Vector2<T>{};
      xn.y() = (x.y() - v0()) / fy();
      xn.x() = (x.x() - u0() - shear() * xn.y()) / fx();
      return xn;
    }

    //! @brief Apply only in the normalized coordinates.
    auto lens_distortion(const Eigen::Vector2<T>& xun) const
        -> Eigen::Vector2<T>
    {
      // Radial term.
      const auto r2 = xun.squaredNorm();
      auto rpowers = Eigen::Vector3<T>{};
      rpowers[0] = r2;
      for (auto i = 1; i < 3; ++i)
        rpowers[i] = rpowers[i - 1] * r2;
      const Eigen::Vector2<T> radial_term = k().dot(rpowers) * xun;

      // Tangential term.
      const Eigen::Matrix2<T> Tmat =
          r2 * Eigen::Matrix2<T>::Identity() + 2 * xun * xun.transpose();
      const Eigen::Vector2<T> tangential_term = Tmat * p();

      return radial_term + tangential_term;
    }

    //! @brief Iterative method to remove distortion.
    auto correct_lens_distortion(const Eigen::Vector2<T>& xd,
                                 int num_iterations = 10, T eps = T(1e-8)) const
        -> Eigen::Vector2<T>
    {
      auto xu = xd;
      for (auto iter = 0; iter < num_iterations &&
                          (xu + lens_distortion(xu) - xd).norm() > eps;
           ++iter)
        xu = xd - lens_distortion(xu);

      return xu;
    }

    auto project(const Eigen::Vector3<T>& ray) const -> Eigen::Vector2<T>
    {
      // To normalized undistorted camera coordinates.
      const Eigen::Vector2<T> xun = ray.hnormalized();
      // To distorted camera coordinats
      const Eigen::Vector2<T> xdn = xun + lens_distortion(xun);
      return to_pixel_coordinates(xdn);
    }

    auto backproject(const Eigen::Vector2<T>& xd) const -> Eigen::Vector3<T>
    {
      // Back to normalized camera coordinates.
      const auto xdn = to_normalized_camera_coordinates(xd);

      // Apply the iterative correction algorithm on the normalized camera
      // coordinates.
      const auto xun = correct_lens_distortion(xdn);

      return xun.homogeneous();
    }

    auto distort(const Eigen::Vector2<T> xu) const -> Eigen::Vector2<T>
    {
      const auto xun = to_normalized_camera_coordinates(xu);
      const Eigen::Vector2<T> xdn = xun + lens_distortion(xun);
      return to_pixel_coordinates(xdn);
    }

    auto undistort(const Eigen::Vector2<T> xd) const -> Eigen::Vector2<T>
    {
      const auto xdn = to_normalized_camera_coordinates(xd);
      const Eigen::Vector2<T> xun = correct_lens_distortion(xdn);
      return to_pixel_coordinates(xun);
    }
  };

  template <typename T>
  using BrownConradyDistortionModelView =
      BrownConradyDistortionModelBase<VectorView<T, 10>>;

  template <typename T>
  using BrownConradyDistortionModelConstView =
      BrownConradyDistortionModelBase<ConstVectorView<T, 10>>;

  template <typename T>
  using BrownConradyDistortionModel =
      BrownConradyDistortionModelBase<Eigen::Vector<T, 10>>;

}  // namespace DO::Sara::v2
