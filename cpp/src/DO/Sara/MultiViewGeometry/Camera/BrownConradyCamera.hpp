// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Pixel/PixelTraits.hpp>

#include <DO/Sara/ImageProcessing/Interpolation.hpp>


namespace DO::Sara {

  template <typename T>
  struct BrownConradyCamera
  {
    //! @brief Types.
    using Vec2 = Eigen::Matrix<T, 2, 1>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;

    using Mat2 = Eigen::Matrix<T, 2, 2>;
    using Mat3 = Eigen::Matrix<T, 3, 3>;

    //! @brief Original image sizes by the camera.
    Vec2 image_sizes;
    //! @brief Pinhole camera parameters.
    Mat3 K;
    //! @brief Cached inverse calibration matrix.
    Mat3 Kinverse_cached = Mat3::Zero();
    //! @brief Radial distortion coefficients.
    Vec3 k;
    //! @brief Tangential distortion coefficients.
    Vec2 p;

    inline auto focal_lengths() const -> Vec2
    {
      return {K(0, 0), K(1, 1)};
    }

    inline auto principal_point() const -> Vec2
    {
      return K.col(2).head(2);
    }

    inline auto field_of_view() const -> Vec2
    {
      const Eigen::Array<T, 2, 1> tg = image_sizes.array() /  //
                                       focal_lengths().array() / 2.;
      return 2. * tg.atan();
    }

    inline auto undistort(const Vec2& xd) const -> Vec2
    {
      const auto f = focal_lengths();
      const auto c = principal_point();

      // Normalized coordinates
      // TODO: use K.inverse() in the shear coefficient is not zero.
      //       and cache it.
      const Vec2 xn = (xd - c).array() / f.array();

      // Radial correction.
      const auto r2 = xn.squaredNorm();
      const auto rpowers = Vec3{r2, std::pow(r2, 2), std::pow(r2, 3)};
      const auto radial = Vec2{k.dot(rpowers) * xn};

      // Tangential correction.
      const Mat2 Tmat = r2 * Mat2::Identity() + 2 * xn * xn.transpose();
      const Vec2 tangential = Tmat * p;

      // Undistorted coordinates.
      const Vec2 xu = xd + ((radial + tangential).array() * f.array()).matrix();

      return xu;
    }

    auto downscale_image_sizes(T factor) -> void
    {
      K.block(0, 0, 2, 3) /= factor;
      image_sizes /= factor;
    }

    template <typename PixelType>
    auto undistort(const ImageView<PixelType>& src,
                   ImageView<PixelType>& dst) const
    {
      const auto& w = dst.width();
      const auto& h = dst.height();

#pragma omp parallel for
      for (auto y = 0; y < h; ++y)
      {
        for (auto x = 0; x < w; ++x)
        {
          const Eigen::Vector2d p = undistort(Vec2(x, y)).template cast<double>();
          const auto in_image_domain = 0 <= p.x() && p.x() < w - 1 &&  //
                                       0 <= p.y() && p.y() < h - 1;
          auto color = interpolate(src, p);
          if constexpr (std::is_same_v<PixelType, Rgb8>)
            color /= 255;

          auto color_converted = PixelType{};
          smart_convert_color(color, color_converted);
          dst(x, y) = in_image_domain  //
                          ? color_converted
                          : PixelTraits<PixelType>::zero();
        }
      }
    }
  };


}  // namespace DO::Sara
