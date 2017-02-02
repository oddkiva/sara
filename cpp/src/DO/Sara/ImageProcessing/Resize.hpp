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

#include <DO/Sara/Core/MultiArray/MultiArray.hpp>
#include <DO/Sara/Core/Image/Operations.hpp>
#include <DO/Sara/ImageProcessing/Deriche.hpp>
#include <DO/Sara/ImageProcessing/Interpolation.hpp>


namespace DO { namespace Sara {

  /*!
    @ingroup MultiArrayProcessing
    @defgroup Scaling Reduce, enlarge, warp image...
    @{
   */

  //! @brief Upscale image.
  template <typename T, int N>
  Image<T, N> upscale(const ImageView<T, N>& src, int fact)
  {
    auto dst = Image<T, N>(src.sizes() * fact);
    for (auto it = dst.begin_array() ; !it.end(); ++it)
      *it = src(it.position() / fact);
    return dst;
  }

  //! @brief Downscale image.
  template <typename T, int N>
  Image<T, N> downscale(const ImageView<T, N>& src, int fact)
  {
    auto dst = Image<T, N>(src.sizes() / fact);
    for (auto it = dst.begin_array(); !it.end(); ++it)
      *it = src(it.position() * fact);
    return dst;
  }

  //! @brief Find min and max coefficient of a vector.
  template <typename T, int N>
  inline std::pair<T, T> range(const Matrix<T, N, 1>& v)
  {
    return { *std::min_element(v.data(), v.data() + N),
             *std::max_element(v.data(), v.data() + N) };
  }

  //! @brief Reduce image.
  template <typename T, int N>
  Image<T, N> reduce(const ImageView<T, N>& src, Matrix<int, N, 1> new_sizes,
                     bool keep_ratio = false)
  {
    // Typedefs.
    using DoublePixel =
        typename PixelTraits<T>::template Cast<double>::pixel_type;
    using ChannelType = typename PixelTraits<T>::channel_type;
    using Cast = typename PixelTraits<T>::template Cast<double>;

    // Convert scalar values to double type.
    auto double_src = src.cwise_transform([](const T& pixel) {
      return Cast::apply(pixel);
    });

    auto original_sizes = src.sizes().template cast<double>();
    auto scale_factors = Matrix<double, N, 1>{
      original_sizes.cwiseQuotient(new_sizes.template cast<double>())
    };
    auto min_max = range(scale_factors);

    if (keep_ratio)
    {
      scale_factors.fill(min_max.second);
      new_sizes = (original_sizes / min_max.second).template cast<int>();
    }

    // Determine the right blurring factor using the following formula.
    auto sigmas = Matrix<double, N, 1>{
      1.5*((scale_factors.array().sqrt() - .99).matrix())
    };

    // Blur with Deriche filter.
    inplace_deriche_blur(double_src, sigmas);

    // Create the new image by interpolating pixel values.
    auto dst = Image<T, N>{ new_sizes };
    auto dst_it = dst.begin_array();
    for ( ; !dst_it.end(); ++dst_it)
    {
      auto position = Matrix<double, N, 1>{ dst_it
        .position()
        .template cast<double>()
        .cwiseProduct(scale_factors)
      };

      auto double_pixel_value = interpolate(double_src, position);
      *dst_it = PixelTraits<DoublePixel>::template Cast<ChannelType>::apply(
          double_pixel_value);
    }

    return dst;
  }

  //! @brief Reduce image.
  template <typename T>
  inline Image<T, 2> reduce(const ImageView<T, 2>& image, int w, int h,
                            bool keep_ratio = false)
  {
    return reduce(image, Vector2i{ w, h }, keep_ratio);
  }

  //! @brief Reduce image.
  template <typename T>
  inline Image<T, 3> reduce(const ImageView<T, 3>& image, int w, int h, int d,
                            bool keep_ratio = false)
  {
    return reduce(image, Vector3i{ w, h, d }, keep_ratio);
  }

  //! @brief Reduce image.
  template <typename T, int N>
  inline Image<T, N> reduce(const ImageView<T, N>& image, double fact)
  {
    Matrix<double, N, 1> new_sizes;
    new_sizes = image.sizes().template cast<double>() / fact;
    return reduce(image, new_sizes.template cast<int>().eval());
  }

  //! @brief Enlarge image.
  template <typename T, int N>
  inline Image<T, N> enlarge(const ImageView<T, N>& image,
                             Matrix<int, N, 1> new_sizes,
                             bool keep_ratio = false)
  {
    // Typedefs.
    using DoublePixel = typename PixelTraits<T>::template Cast<double>::pixel_type;
    using ChannelType = typename PixelTraits<T>::channel_type;

    // Determine the right blurring factor.
    auto original_sizes = image.sizes().template cast<double>();
    auto scale_factor = Matrix<double, N, 1>{
      original_sizes.cwiseQuotient(new_sizes.template cast<double>())
    };
    auto min_max = range(scale_factor);

    if (keep_ratio)
    {
      scale_factor.fill(min_max.second);
      new_sizes = (original_sizes / min_max.second).template cast<int>();
    }

    // Create the new image by interpolation.
    auto dst = Image<T, N>{ new_sizes };

    for (auto dst_it = dst.begin_array(); !dst_it.end(); ++dst_it)
    {
      auto position = Matrix<double, N, 1>{ dst_it.position()
        .template cast<double>()
        .cwiseProduct(scale_factor)
      };

      auto double_pixel_value = interpolate(image, position);

      *dst_it = PixelTraits<DoublePixel>::template Cast<ChannelType>::apply(
          double_pixel_value);
    }
    return dst;
  }

  //! @brief Enlarge image.
  template <typename T>
  inline Image<T, 2> enlarge(const ImageView<T, 2>& image,
                             int w, int h, bool keep_ratio = false)
  {
    return enlarge(image, Point2i(w,h), keep_ratio);
  }

  //! @brief Enlarge image.
  template <typename T>
  inline Image<T, 3> enlarge(const ImageView<T, 3>& image, int w, int h, int d,
                             bool keep_ratio = false)
  {
    return enlarge(image, Vector3i(w,h,d), keep_ratio);
  }

  //! @brief Enlarge image.
  template <typename T, int N>
  inline Image<T, N> enlarge(const ImageView<T, N>& image, double fact)
  {
    auto new_sizes = image.sizes().template cast<double>() * fact;
    return enlarge(image, new_sizes.template cast<int>().eval());
  }

  //! @}

} /* namespace Sara */
} /* namespace DO */
