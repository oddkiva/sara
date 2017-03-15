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


namespace DO { namespace Sara {

  template <typename T, typename S>
  void warp(const ImageView<T>& src, ImageView<T>& dst,
            const Matrix<S, 3, 3>& homography_from_dst_to_src,
            const T& default_fill_color = PixelTraits<T>::min())
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
        H_P.x() >= 0 && H_P.x() < S(src.width()) &&
        H_P.y() >= 0 && H_P.y() < S(src.height());

      // Fill with either the default value or the interpolated value.
      if (position_is_in_src_domain)
      {
        Vector2d H_p{ H_P.template head<2>().template cast<double>() };
        auto pixel_value = interpolate(src, H_p);
        *it = PixelTraits<DoublePixel>::template Cast<ChannelType>::apply(
          pixel_value);
      }
      else
        *it = default_fill_color;
    }
  }

} /* namespace Sara */
} /* namespace DO */
