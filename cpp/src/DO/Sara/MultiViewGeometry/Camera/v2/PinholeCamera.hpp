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
  struct PinholeCameraBase : CameraIntrinsicBase<Array>
  {
    using base_type = CameraIntrinsicBase<Array>;
    using base_type::array_type;
    using base_type::data;
    using T = typename Array::value_type;

    // clang-format off
    enum Index
    {
      FX = 0,
      FY = 1,
      S  = 2,
      U0 = 3,
      V0 = 4
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

    auto focal_lengths() -> VectorView<T, 2>
    {
      return VectorView<T, 2>{data.data() + FX};
    }

    auto principal_point() -> VectorView<T, 2>
    {
      return VectorView<T, 2>{data.data() + U0};
    }

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

    auto focal_lengths() const -> ConstVectorView<T, 2>
    {
      return ConstVectorView<T, 2>{data.data() + FX};
    }


    auto principal_point() const -> ConstVectorView<T, 2>
    {
      return ConstVectorView<T, 2>{data.data() + U0};
    }

    auto calibration_matrix() const -> Eigen::Matrix3<T>
    {
      // clang-format off
      const auto K = (Eigen::Matrix3<T>{} <<
        fx(), shear(), u0(),
           0,    fy(), v0(),
           0,       0,    1
      ).finished();
      // clang-format on
      return K;
    }

    //! @brief Project a 3D scene point/ray expressed in the camera frame to the
    //! distorted image coordinates.
    auto project(const Eigen::Vector3<T>& Xc) const -> Eigen::Vector2<T>
    {
      const Eigen::Vector2<T> Xn = Xc.hnormalized();
      return {
          fx() * Xn.x() + shear() * Xn.y() + u0(),  //
          /*              */ fy() * Xn.y() + v0()   //
      };
    }

    //! @brief Backproject a 2D pixel coordinates back to the corresponding 3D
    //! light ray.
    inline auto backproject(const Eigen::Vector2<T>& uv) const
        -> Eigen::Vector3<T>
    {
      // Back to normalized camera coordinates.
      const auto& u = uv.x();
      const auto& v = uv.y();
      const auto y = (v - v0()) / fy();
      const auto x = (u - u0() - shear() * y) / fx();
      static constexpr auto one = static_cast<T>(1);
      return Eigen::Vector3<T>{x, y, one};
    }
  };

  template <typename T>
  using PinholeCameraView = PinholeCameraBase<VectorView<T, 5>>;

  template <typename T>
  using PinholeCameraConstView = PinholeCameraBase<ConstVectorView<T, 5>>;

  template <typename T>
  using PinholeCamera = PinholeCameraBase<Eigen::Vector<T, 5>>;

}  // namespace DO::Sara::v2
