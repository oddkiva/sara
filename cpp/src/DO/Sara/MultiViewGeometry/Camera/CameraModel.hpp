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

#include <DO/Sara/MultiViewGeometry/Camera/PinholeCamera.hpp>


namespace DO::Sara {

  template <typename T>
  class CameraModel
  {
  public:
    using vector2_type = Eigen::Matrix<T, 2, 1>;
    using vector3_type = Eigen::Matrix<T, 3, 1>;
    using matrix3_type = Eigen::Matrix<T, 3, 3>;

    using Vector2 = Eigen::Matrix<T, 2, 1>;
    using Vector3 = Eigen::Matrix<T, 3, 1>;
    using Matrix3 = Eigen::Matrix<T, 3, 3>;

    template <typename MathematicalModel>
    inline CameraModel(MathematicalModel model)
      : _self{new CameraModelType<MathematicalModel>(std::move(model))}
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

    inline auto set_calibration_matrix(const Matrix3& K)
    {
      _self->set_calibration_matrix(K);
    }

    friend inline auto project(const CameraModel& c, const Vector3& x)
        -> Vector2
    {
      return c._self->project(x);
    }

    friend inline auto backproject(const CameraModel& c,
                                   const Vector2& x)
        -> Vector3
    {
      return c._self->backproject(x);
    }

    inline auto calibration_matrix() const -> const Matrix3&
    {
      return _self->calibration_matrix();
    }

    inline auto inverse_calibration_matrix() const -> const Matrix3&
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

      virtual auto set_calibration_matrix(const Matrix3&) -> void = 0;

      virtual auto calibration_matrix() const -> const Matrix3& = 0;

      virtual auto inverse_calibration_matrix() -> Matrix3& = 0;

      virtual auto inverse_calibration_matrix() const -> const Matrix3& = 0;
    };

    template <typename MathematicalModel>
    struct CameraModelType : CameraModelConcept
    {
      static_assert(std::is_same_v<T, typename MathematicalModel::scalar_type>);

      CameraModelType(MathematicalModel model)
        : _model{std::move(model)}
      {
      }

      inline auto copy() const -> CameraModelConcept* override
      {
        return new CameraModelType{*this};
      }

      inline auto project(const Vector3& x) const -> Vector2 override
      {
        return _model.project(x);
      }

      inline auto backproject(const Vector2& x) const -> Vector3 override
      {
        return _model.backproject(x);
      }

      inline auto distort(const Vector2& x) const -> Vector2 override
      {
        if constexpr (std::is_same_v<MathematicalModel, PinholeCamera<T>>)
          return x;
        else
          return _model.distort(x);
      }

      inline auto undistort(const Vector2& x) const -> Vector2 override
      {
        if constexpr (std::is_same_v<MathematicalModel, PinholeCamera<T>>)
          return x;
        else
          return _model.undistort(x);
      }

      inline auto set_calibration_matrix(const Matrix3& K) -> void override
      {
        _model.set_calibration_matrix(K);
      }

      inline auto calibration_matrix() const -> const Matrix3& override
      {
        return _model.K;
      }

      inline auto inverse_calibration_matrix() -> Matrix3& override
      {
        return _model.K_inverse;
      }

      inline auto inverse_calibration_matrix() const -> const Matrix3& override
      {
        return _model.K_inverse;
      }

      MathematicalModel _model;
    };

    std::unique_ptr<CameraModelConcept> _self;
  };


  template <typename CameraModelType, typename PixelType>
  auto undistort(const CameraModelType& camera,
                 const ImageView<PixelType>& src,
                 ImageView<PixelType>& dst)
  {
    using vector2_type = typename CameraModelType::vector2_type;

    const auto& w = dst.width();
    const auto& h = dst.height();

#pragma omp parallel for
    for (auto yx = 0; yx < h * w; ++yx)
    {
      const auto y = yx / w;
      const auto x = yx - y * w;

      const Eigen::Vector2d p =
          camera.distort(vector2_type(x, y)).template cast<double>();

      const auto in_image_domain = 0 <= p.x() && p.x() < src.width() - 1 &&
                                   0 <= p.y() && p.y() < src.height() - 1;
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

  template <typename CameraModelType, typename PixelType>
  auto distort(const CameraModelType& camera,
               const ImageView<PixelType>& src,
               ImageView<PixelType>& dst)
  {
    using vector2_type = typename CameraModelType::vector2_type;

    const auto& w = dst.width();
    const auto& h = dst.height();

#pragma omp parallel for
    for (auto yx = 0; yx < h * w; ++yx)
    {
      const auto y = yx / w;
      const auto x = yx - y * w;
      const Eigen::Vector2d p = camera
                                    .undistort(vector2_type(x, y))
                                    .template cast<double>();
      const auto in_image_domain = 0 <= p.x() && p.x() < src.width() - 1 &&
                                   0 <= p.y() && p.y() < src.height() - 1;
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
