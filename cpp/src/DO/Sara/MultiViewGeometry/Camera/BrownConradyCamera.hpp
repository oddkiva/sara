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
    using Vec9 = Eigen::Matrix<T, 5, 1>;

    using Mat2 = Eigen::Matrix<T, 2, 2>;
    using Mat3 = Eigen::Matrix<T, 3, 3>;

    //! @brief Original image sizes by the camera.
    Vec2 image_sizes;
    //! @brief Pinhole camera parameters.
    Mat3 K;
    //! @brief Radial distortion coefficients.
    Vec3 k;
    //! @brief Tangential distortion coefficients.
    Vec2 p;

    //! @brief Cached inverse calibration matrix.
    mutable std::optional<Mat3> K_inverse;

    //! @brief Cached variable for the inverse distortion calculation.
    //! cf. publication:
    //!   An Exact Formula for Calculating Inverse Radial Lens Distortions,
    //!   Drap and Lefevre.
    Vec9 k_inverse;

    inline auto cache_inverse_calibration_matrix() const
    {
      K_inverse = K.inverse();
    }

    inline auto calculate_drap_lefevre_inverse_coefficients()
    {
      k_inverse(0) = -k(0);
      k_inverse(1) = 3 * std::pow(k(0), 2) - k(1);
      k_inverse(2) = 8 * k(0) * k(1) - 12 * std::pow(k(1), 3) - k(2);

      const auto k3 = 0;
      k_inverse(3) =   55 * std::pow(k(0), 4)         //
                     - 55 * std::pow(k(0), 2) * k(1)  //
                     + 5 * std::pow(k(1), 2)          //
                     + 10 * k(0) * k(2)               //
                     - k3;
      k_inverse(4) = -273 * std::pow(k(0), 5) //
                     +364 * std::pow(k(0), 3) * k(1) //
                     -78 * k(0) * std::pow(k(1), 2) //
                     -78 * std::pow(k(0), 2) * k(2) //
                     +12 * k(1) * k(2) //
                     +12 * k(0) * k3;
      // for (int i = 4; i < 9; ++i)
      //   k_inverse(i) = 0;
    }

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

    //! @brief Using Drap-Lefevre method.
    //! Valid only if the tangential coefficients are zero.
    inline auto undistort(const Vec2& xu) const -> Vec2
    {
      if (!K_inverse.has_value())
        cache_inverse_calibration_matrix();
      const Vec2 xun = (K_inverse.value() * xu.homogeneous()).head(2);

      const auto r2 = xun.squaredNorm();
      auto rpowers = Vec9{};
      rpowers <<
        r2,
        std::pow(r2, 2),
        std::pow(r2, 3),
        std::pow(r2, 4),
        std::pow(r2, 5);
      const auto radial = k_inverse.dot(rpowers);

      const Vec2 xdn = xun * (1 + radial);

      const Vec2 xd = (K * xdn.homogeneous()).head(2);

      return xd;
    }

    inline auto undistort_with_bisection(const Vec2& xu)  const -> Vec2
    {
      static constexpr auto max_iter = 10;
      static constexpr auto eps = 1e-6f;

      if (!K_inverse.has_value())
        cache_inverse_calibration_matrix();
      const Vec2 xun = (K_inverse.value() * xu.homogeneous()).head(2);
      const auto ru = xun.norm();

      // The radial distortion function.
      auto f = [this](float r_distorted) {
        const auto& rd = r_distorted;
        const auto rd2 = rd * rd;
        const auto rd3 = rd2 * rd;
        const auto rd5 = rd3 * rd2;
        return rd + k(0) * rd3 + k(1) * rd5;
      };

      // Calculate the bounds of r_distorted.
      auto r_lower = 0.f;
      auto r_upper = 2 * ru;
      auto f_r_distorted = f(r_upper);
      while (f_r_distorted < ru)
      {
        r_lower = r_upper;
        r_upper = 2 * r_upper;
      }

      // THE UNKNOWN.
      auto r_dist_estimate = 0.5f * (r_lower + r_upper);
      auto f_at_r_dist_estimate = f(r_dist_estimate);

      // By monotonicity.
      auto iter = 0;
      while (std::abs(f_at_r_dist_estimate - ru) > eps &&
             iter < max_iter)
      {
        if (f_at_r_dist_estimate > ru)
          r_upper = r_dist_estimate;
        else
          r_lower = r_dist_estimate;

        r_dist_estimate = 0.5 * (r_lower + r_upper);
        f_at_r_dist_estimate = f(r_dist_estimate);

        ++iter;
      }

      const Vec2 xdn = (r_dist_estimate / ru) * xun;
      const Vec2 xd = (K * xdn.homogeneous()).hnormalized();

      return xd;
    }

    inline auto distort(const Vec2& xd) const -> Vec2
    {
      // Normalized coordinates.
      if (!K_inverse.has_value())
        cache_inverse_calibration_matrix();
      const Vec2 xdn = (K_inverse.value() * xd.homogeneous()).head(2);

      // Radial correction.
      const auto r2 = xdn.squaredNorm();
      const auto r4 = r2 * r2;
      const auto r6 = r4 * r2;
      const auto rpowers = Vec3{r2, r4, r6};
      const auto radial = Vec2{k.dot(rpowers) * xdn};

      // Tangential correction.
      const Mat2 Tmat = r2 * Mat2::Identity() + 2 * xdn * xdn.transpose();
      const Vec2 tangential = Tmat * p;

      const Vec2 xun = xdn + radial + tangential;

      // Go back to pixel coordinates.
      const Vec2 xu = (K * xun.homogeneous()).head(2);

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
      for (auto yx = 0; yx < h * w; ++yx)
      {
        const auto y = yx / w;
        const auto x = yx - y * w;
        const Eigen::Vector2d p = distort(Vec2(x, y)).template cast<double>();

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
    auto distort(const ImageView<PixelType>& src,
                 ImageView<PixelType>& dst) const
    {
      const auto& w = dst.width();
      const auto& h = dst.height();

#pragma omp parallel for
      for (auto yx = 0; yx < h * w; ++yx)
      {
        const auto y = yx / w;
        const auto x = yx - y * w;
        const Eigen::Vector2d p = undistort_with_bisection(Vec2(x, y))  //
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
  };


}  // namespace DO::Sara
