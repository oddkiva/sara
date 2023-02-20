// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/ImageProcessing/Interpolation.hpp>


namespace DO::Sara {

  template <typename T, typename S>
  inline auto warp(const ImageView<T>& src, ImageView<T>& dst,
                   const Matrix<S, 3, 3>& homography_from_dst_to_src,
                   const T& default_fill_color = PixelTraits<T>::min()) -> void
  {
    using DoublePixel =
        typename PixelTraits<T>::template Cast<double>::pixel_type;
    using ChannelType = typename PixelTraits<T>::channel_type;
    using Vector3 = Matrix<S, 3, 1>;

    const auto& H = homography_from_dst_to_src;

    for (auto it = dst.begin_array(); !it.end(); ++it)
    {
      // Get the corresponding coordinates in the source image.
      auto H_P = Vector3{};
      H_P = H * (Vector3() << it.position().template cast<S>(), 1).finished();
      H_P /= H_P(2);

      // Check if the position is not in the src domain [0,w[ x [0,h[.
      auto position_is_in_src_domain =
          H_P.x() >= 0 && H_P.x() < S(src.width()) && H_P.y() >= 0 &&
          H_P.y() < S(src.height());

      // Fill with either the default value or the interpolated value.
      if (position_is_in_src_domain)
      {
        Vector2d H_p{H_P.template head<2>().template cast<double>()};
        auto pixel_value = interpolate(src, H_p);
        *it = PixelTraits<DoublePixel>::template Cast<ChannelType>::apply(
            pixel_value);
      }
      else
        *it = default_fill_color;
    }
  }


  inline auto warp(const ImageView<float>& u_map,  //
                   const ImageView<float>& v_map,  //
                   const ImageView<float>& frame,
                   ImageView<float>& frame_warped) -> void
  {
    const auto w = frame.width();
    const auto h = frame.height();
    const auto wh = w * h;

#pragma omp parallel for
    for (auto p = 0; p < wh; ++p)
    {
      // Destination pixel.
      const auto y = p / w;
      const auto x = p - w * y;

      auto xyd = Eigen::Vector2d{};
      xyd << u_map(x, y), v_map(x, y);

      const auto in_image_domain = 0 <= xyd.x() && xyd.x() < w - 1 &&  //
                                   0 <= xyd.y() && xyd.y() < h - 1;
      if (!in_image_domain)
      {
        frame_warped(x, y) = 0.f;
        continue;
      }

      const auto color = interpolate(frame, xyd);
      frame_warped(x, y) = color;
    }
  }

  inline auto warp(const ImageView<float>& u_map,  //
                   const ImageView<float>& v_map,  //
                   const ImageView<Rgb8>& frame,   //
                   ImageView<Rgb8>& frame_warped) -> void
  {
    const auto w = frame.width();
    const auto h = frame.height();
    const auto wh = w * h;

#pragma omp parallel for
    for (auto p = 0; p < wh; ++p)
    {
      // Destination pixel.
      const auto y = p / w;
      const auto x = p - w * y;

      auto xyd = Eigen::Vector2d{};
      xyd << u_map(x, y), v_map(x, y);

      const auto in_image_domain = 0 <= xyd.x() && xyd.x() < w - 1 &&  //
                                   0 <= xyd.y() && xyd.y() < h - 1;
      if (!in_image_domain)
      {
        frame_warped(x, y) = Black8;
        continue;
      }

      auto color = interpolate(frame, xyd);
      color /= 255;

      auto color_converted = Rgb8{};
      smart_convert_color(color, color_converted);

      frame_warped(x, y) = color_converted;
    }
  }

}  // namespace DO::Sara
