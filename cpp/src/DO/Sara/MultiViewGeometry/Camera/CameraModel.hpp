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

#include <iostream>
#include <memory>

#include <Eigen/Core>


namespace DO::Sara {

  template <typename T>
  struct PinholeCamera;


  template <typename T>
  class CameraModel
  {
  public:
    using Vector2 = Eigen::Matrix<T, 2, 1>;
    using Vector3 = Eigen::Matrix<T, 3, 1>;
    using Matrix3 = Eigen::Matrix<T, 3, 3>;

    template <typename Impl>
    inline CameraModel(Impl impl)
      : _self{new CameraModelImpl<Impl>(std::move(impl))}
    {
    }

    inline CameraModel(const CameraModel& c)
      : _self{c._self->copy()}
    {
    }

    inline CameraModel(CameraModel&&) noexcept = default;

    inline CameraModel& operator=(const CameraModel& c)
    {
      auto tmp = CameraModel{c};
      *this = std::move(tmp);
      return *this;
    }

    inline CameraModel& operator=(CameraModel&& c) = default;

    friend inline auto project(const CameraModel& c,
                               const Eigen::Matrix<T, 3, 1>& x)
        -> Eigen::Matrix<T, 2, 1>
    {
      return c._self->project(x);
    }

    friend inline auto backproject(const CameraModel& c,
                                   const Eigen::Matrix<T, 2, 1>& x)
        -> Eigen::Matrix<T, 3, 1>
    {
      return c._self->backproject(x);
    }

  private:
    struct CameraModelConcept
    {
      virtual ~CameraModelConcept() = default;

      virtual CameraModelConcept* copy() const = 0;

      virtual auto distort(const Vector2&) const -> Vector2 = 0;

      virtual auto undistort(const Vector2&) const -> Vector2 = 0;

      virtual auto project(const Vector3&) const -> Vector2 = 0;

      virtual auto backproject(const Vector2&) const -> Vector3 = 0;

      virtual auto calibration_matrix() -> Matrix3& = 0;

      virtual auto calibration_matrix() const -> const Matrix3& = 0;

      virtual auto inverse_calibration_matrix() -> Eigen::Matrix3f& = 0;

      virtual auto inverse_calibration_matrix() const -> const Matrix3& = 0;
    };

    template <typename Impl>
    struct CameraModelImpl : CameraModelConcept
    {
      static_assert(std::is_same_v<T, typename Impl::scalar_type>);

      CameraModelImpl(Impl impl)
        : _impl{std::move(impl)}
      {
      }

      inline auto copy() const -> CameraModelConcept* override
      {
        return new CameraModelImpl{*this};
      }

      inline auto project(const Vector3& x) const -> Vector2 override
      {
        return _impl.project(x);
      }

      inline auto backproject(const Vector2& x) const -> Vector3 override
      {
        return _impl.backproject(x);
      }

      inline auto distort(const Vector2& x) const -> Vector2 override
      {
        if constexpr (std::is_same_v<Impl, PinholeCamera<T>>)
          return x;
        else
          return _impl.distort(x);
      }

      inline auto undistort(const Vector2& x) const -> Vector2 override
      {
        if constexpr (std::is_same_v<Impl, PinholeCamera<T>>)
          return x;
        else
          return _impl.undistort(x);
      }

      inline auto calibration_matrix() -> Matrix3& override
      {
        return _impl.K;
      }

      inline auto calibration_matrix() const -> const Matrix3& override
      {
        return _impl.K;
      }

      inline auto inverse_calibration_matrix() -> Matrix3& override
      {
        return _impl.K_inverse;
      }

      inline auto inverse_calibration_matrix() const -> const Matrix3& override
      {
        return _impl.K_inverse;
      }

      Impl _impl;
    };

    std::unique_ptr<CameraModelConcept> _self;
  };

}  // namespace DO::Sara
