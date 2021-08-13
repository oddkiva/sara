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

#include <DO/Sara/Core/Image.hpp>


namespace DO::Sara {

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

    inline auto set_calibration_matrix(const Eigen::Matrix<T, 3, 3>& K)
    {
      _self->set_calibration_matrix(K);
    }

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

    inline auto calibration_matrix() const -> const Eigen::Matrix<T, 3, 3>&
    {
      return _self->calibration_matrix();
    }

    inline auto inverse_calibration_matrix() const
        -> const Eigen::Matrix<T, 3, 3>&
    {
      return _self->inverse_calibration_matrix();
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

      virtual auto set_calibration_matrix(const Eigen::Matrix3f&) -> void = 0;

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

      inline auto set_calibration_matrix(const Matrix3& K) -> void override
      {
        _impl.K = K;
        _impl.K_inverse = K.inverse();
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


  template <typename PixelType>
  auto undistort(const CameraModel& camera, const ImageView<PixelType>& src,
                 ImageView<PixelType>& dst)
  {
    const auto& w = dst.width();
    const auto& h = dst.height();

#pragma omp parallel for
    for (auto yx = 0; yx < h * w; ++yx)
    {
      const auto y = yx / w;
      const auto x = yx - y * w;
      const Eigen::Vector2d p =
          camera.distort(vector2_type(x, y)).template cast<double>();

      const auto in_image_domain = 0 <= p.x() && p.x() < w - 1 &&  //
                                   0 <= p.y() && p.y() < h - 1;
      if (!in_image_domain)
      {
        dst(x, y) = PixelTraits<PixelType>::zero();
        continue;
      }

      auto color = interpolate(src, p);
      if constexpr (std::is_same_v<PixelType, Rgb8>)
        color /= 255;

      auto color_converted = PixelType{};
      smart_convert_color(color, color_converted);
      dst(x, y) = color_converted;
    }
  }

  template <typename PixelType>
  auto distort(const CameraModel& camera, const ImageView<PixelType>& src,
               ImageView<PixelType>& dst)
  {
    const auto& w = dst.width();
    const auto& h = dst.height();

#pragma omp parallel for
    for (auto yx = 0; yx < h * w; ++yx)
    {
      const auto y = yx / w;
      const auto x = yx - y * w;
      const Eigen::Vector2d p = camera
                                    .undistort(vector2_type(x, y))  //
                                    .template cast<double>();
      const auto in_image_domain = 0 <= p.x() && p.x() < w - 1 &&  //
                                   0 <= p.y() && p.y() < h - 1;
      if (!in_image_domain)
      {
        dst(x, y) = PixelTraits<PixelType>::zero();
        continue;
      }

      auto color = interpolate(src, p);
      if constexpr (std::is_same_v<PixelType, Rgb8>)
        color /= 255;

      auto color_converted = PixelType{};
      smart_convert_color(color, color_converted);
      dst(x, y) = color_converted;
    }
  }

}  // namespace DO::Sara
