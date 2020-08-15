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

#include <DO/Sara/Core/PixelTraits.hpp>

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

    auto undistort(const Vec2& xd) const -> Vec2
    {
      const auto f = focal_lengths();
      const auto c = principal_point();

      // Normalized coordinates
      const Vec2 xn = (xd - c).array() / f.array();

      // Radial correction.
      const auto r2 = xn.squaredNorm();
      const auto rpowers = Vec3{r2, pow(r2, T(2)), pow(r2, T(3))};
      const auto radial = Vec2{k.dot(rpowers) * xn};

      // Tangential correction.
      auto Tx = Mat2{};
      Tx << 3 * p(0), p(1),
                p(1), p(0);
      auto Ty = Mat2{};
      Ty << p(1),     p(0),
            p(0), 3 * p(1);
      const Vec2 tangential = {xn.transpose() * Tx * xn,
                               xn.transpose() * Ty * xn};

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

#pragma omp parallel for
      for (auto y = 0; y < dst.height(); ++y)
      {
        for (auto x = 0; x < dst.width(); ++x)
        {
          const auto p = intrinsics.undistort(Vec2(x, y));
          const auto in_image_domain = 0 <= p.x() && p.x() < w - 1 &&  //
                                       0 <= p.y() && p.y() < h - 1;
          dst(y, x) = in_image_domain ?  //
            interpolate(src, p) : PixelTraits<Pixel>::template zero();
        }
      }
    }
  };


}  // namespace DO::Sara
