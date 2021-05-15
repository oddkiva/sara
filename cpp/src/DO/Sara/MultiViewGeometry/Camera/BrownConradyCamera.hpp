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

#include <DO/Sara/MultiViewGeometry/Camera/PinholeCamera.hpp>


namespace DO::Sara {

  template <typename T>
  struct BrownConradyCamera: PinholeCamera<T>
  {
    //! @brief Types.
    using base_type = PinholeCamera<T>;
    using vector2_type = typename base_type::vector2_type;
    using vector3_type = typename base_type::vector3_type;
    using matrix2_type = Eigen::Matrix<T, 2, 2>;
    using matrix3_type = typename base_type::matrix3_type;

    using base_type::image_sizes;
    using base_type::K;
    using base_type::K_inverse;

    //! @brief Radial distortion coefficients.
    vector3_type k;
    //! @brief Tangential distortion coefficients.
    vector2_type p;

    inline auto undistort(const vector2_type&)  const -> vector2_type
    {
      throw std::runtime_error{"Not Implemented!"};
      return {};
    }

    inline auto distort(const vector2_type& xd) const -> vector2_type
    {
      // Normalized coordinates.
      const vector2_type xdn = (K_inverse * xd.homogeneous()).head(2);

      // Radial correction.
      const auto r2 = xdn.squaredNorm();
      const auto r4 = r2 * r2;
      const auto r6 = r4 * r2;
      const auto rpowers = vector3_type{r2, r4, r6};
      const auto radial = vector2_type{k.dot(rpowers) * xdn};

      // Tangential correction.
      const matrix2_type Tmat = r2 * matrix2_type::Identity() + 2 * xdn * xdn.transpose();
      const vector2_type tangential = Tmat * p;

      const vector2_type xun = xdn + radial + tangential;

      // Go back to pixel coordinates.
      const vector2_type xu = (K * xun.homogeneous()).head(2);

      return xu;
    }

    auto downscale_image_sizes(T factor) -> void
    {
      K.block(0, 0, 2, 3) /= factor;
      K_inverse = K.inverse();
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
        const Eigen::Vector2d p = distort(vector2_type(x, y)).template cast<double>();

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
        const Eigen::Vector2d p = undistort(vector2_type(x, y))  //
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
