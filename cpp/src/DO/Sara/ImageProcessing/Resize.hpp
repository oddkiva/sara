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

#include <DO/Sara/Core/Image/Operations.hpp>
#include <DO/Sara/Core/MultiArray/MultiArray.hpp>
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
    for (auto it = dst.begin_array(); !it.end(); ++it)
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
    return {*std::min_element(v.data(), v.data() + N),
            *std::max_element(v.data(), v.data() + N)};
  }


  //! @{
  //! @brief Reduce image.
  template <typename T, int N>
  void reduce(const ImageView<T, N>& src, ImageView<T, N>& dst)
  {
    if (dst.sizes() != dst.sizes().cwiseMin(src.sizes()))
      throw std::range_error{"The destination image must have smaller sizes "
                             "than the source image!"};

    if (dst.sizes().minCoeff() <= 0)
      throw std::range_error{
          "The sizes of the destination image must be positive!"};

    // Typedefs.
    using DoublePixel =
        typename PixelTraits<T>::template Cast<double>::pixel_type;
    using ChannelType = typename PixelTraits<T>::channel_type;
    using Cast = typename PixelTraits<T>::template Cast<double>;

    // Convert scalar values to double type.
    auto src_d =
        src.cwise_transform([](const T& pixel) { return Cast::apply(pixel); });

    const auto size_ratio = src.sizes().template cast<double>().array() /
                            dst.sizes().template cast<double>().array();

    // Determine the right blurring factor using the following formula.
    const Matrix<double, N, 1> sigmas = 1.5 * (size_ratio.sqrt() - .99);

    // Blur with Deriche filter.
    for (int i = 0; i < N; ++i)
    {
      // Don't apply the blur filter in this dimension.
      if (src.size(i) == dst.size(i))
        continue;

      inplace_deriche(src_d, sigmas[i], 0, i);
    }

    // Create the new image by interpolating pixel values.
    auto dst_it = dst.begin_array();
    for (; !dst_it.end(); ++dst_it)
    {
      const Matrix<double, N, 1> position =
          dst_it.position().template cast<double>().array() * size_ratio;

      const auto double_value = interpolate(src_d, position);
      *dst_it = PixelTraits<DoublePixel>::template Cast<ChannelType>::apply(
          double_value);
    }
  }

  template <typename T, int N>
  Image<T, N> reduce(const ImageView<T, N>& src, Matrix<int, N, 1> new_sizes,
                     bool keep_ratio = false)
  {
    const Matrix<double, N, 1> original_sizes =
        src.sizes().template cast<double>();
    Matrix<double, N, 1> scale_factors =
        original_sizes.cwiseQuotient(new_sizes.template cast<double>());
    const auto min_max = range(scale_factors);

    if (keep_ratio)
    {
      scale_factors.fill(min_max.second);
      new_sizes = (original_sizes / min_max.second).template cast<int>();
    }

    // Create the new image by interpolating pixel values.
    auto dst = Image<T, N>{new_sizes};
    reduce(src, dst);
    return dst;
  }

  template <typename T, int N>
  inline Image<T, N> reduce(const ImageView<T, N>& image, double fact)
  {
    Matrix<double, N, 1> new_sizes;
    new_sizes = image.sizes().template cast<double>() / fact;
    return reduce(image, new_sizes.template cast<int>().eval());
  }
  //! @}


  //! @{
  //! @brief Enlarge image.
  template <typename T, int N>
  void enlarge(const ImageView<T, N>& src, ImageView<T, N>& dst)
  {
    if (dst.sizes() != dst.sizes().cwiseMax(src.sizes()))
      throw std::range_error{"The destination image must have smaller sizes "
                             "than the source image!"};

    if (dst.sizes().minCoeff() <= 0)
      throw std::range_error{
          "The sizes of the destination image must be positive!"};

    // Typedefs.
    using DoublePixel =
        typename PixelTraits<T>::template Cast<double>::pixel_type;
    using ChannelType = typename PixelTraits<T>::channel_type;

    const auto size_ratio = src.sizes().template cast<double>().array() /
                            dst.sizes().template cast<double>().array();

    // Create the new image by interpolation.
    for (auto dst_it = dst.begin_array(); !dst_it.end(); ++dst_it)
    {
      const Matrix<double, N, 1> position =
          dst_it.position().template cast<double>().array() * size_ratio;

      const auto double_pixel_value = interpolate(src, position);

      *dst_it = PixelTraits<DoublePixel>::template Cast<ChannelType>::apply(
          double_pixel_value);
    }
  }

  template <typename T, int N>
  inline Image<T, N> enlarge(const ImageView<T, N>& image,
                             Matrix<int, N, 1> new_sizes,
                             bool keep_ratio = false)
  {
    // Determine the right blurring factor.
    const auto original_sizes = image.sizes().template cast<double>();
    Matrix<double, N, 1> scale_factor =
        original_sizes.cwiseQuotient(new_sizes.template cast<double>());
    const auto min_max = range(scale_factor);

    if (keep_ratio)
    {
      scale_factor.fill(min_max.second);
      new_sizes = (original_sizes / min_max.second).template cast<int>();
    }

    auto dst = Image<T, N>{new_sizes};
    enlarge(image, dst);

    return dst;
  }

  template <typename T, int N>
  inline Image<T, N> enlarge(const ImageView<T, N>& image, double fact)
  {
    const auto new_sizes = image.sizes().template cast<double>() * fact;
    return enlarge(image, new_sizes.template cast<int>().eval());
  }
  //! @}


  //! @{
  //! @brief Resize the image.
  template <typename T, int N>
  void resize(const ImageView<T, N>& src, ImageView<T, N>& dst)
  {
    const auto size_ratio = dst.sizes().template cast<double>().array() /
                            src.sizes().template cast<double>().array();

    const auto min_ratio = size_ratio.minCoeff();
    const auto max_ratio = size_ratio.maxCoeff();

    if (max_ratio < 1.f)
      reduce(src, dst);
    else if (min_ratio >= 1.f)
      enlarge(src, dst);
    else  // min_ratio < 1.f && max_ratio >= 1.f
    {
      const Matrix<int, N, 1> enlarged_src_sizes =
          src.sizes().cwiseMax(dst.sizes());
      const auto enlarged_src = enlarge(src, enlarged_src_sizes);
      reduce(enlarged_src, dst);
    }
  }

  template <typename T, int N>
  auto resize(const ImageView<T, N>& src, const Vector2i& sizes) -> Image<T, N>
  {
    auto dst = Image<T, N>{sizes};
    resize(src, dst);
    return dst;
  }
  //! @}

  // @brief Image resizing functor that preserves the image size ratio.
  struct SizeRatioPreservingImageResizer
  {
    using Window = RowVector4i;
    using Padding = RowVector4i;
    using Scale = float;

    template <typename T>
    auto operator()(const ImageView<T>& src, ImageView<T>& dst,
                    int dst_min_size = 0) const  //
        -> std::tuple<Window, Scale, Padding>
    {
      const auto dst_max_size = dst.sizes().maxCoeff();

      // Determine the scale of the resized image. The original image is at
      // scale 1.
      auto scale = float{};
      {
        // Scale up the image if the image sizes are smaller than the input
        // tensor sizes.
        //
        // Make sure that if the image is too small, we don't scale up the image
        // sizes up to the `dst_max_size = 1024`. Hence we put a limit of
        // `dst_min_size = 800` instead of `1024`.
        if (dst_min_size > 0)
          scale = std::max(1.f, float(dst_min_size) / src.sizes().minCoeff());
        else
        {
          scale = (dst.sizes().template cast<double>().array() /
                   src.sizes().template cast<double>().array())
                      .maxCoeff();
        }

        // Make sure that the resized image still fits within the input
        // tensor.
        auto src_max_size = src.sizes().maxCoeff();
        if (std::round(src_max_size * scale) > dst_max_size)
          scale = float(dst_max_size) / src_max_size;
      }

      const Eigen::Vector2i resized_sizes =
          (src.sizes().template cast<float>() * scale)
              .array()
              .round()
              .template cast<int>();
      const auto resized_image = resize(src, resized_sizes);

      // Determine the padding size of the image.
      const auto resized_h = resized_image.height();
      const auto resized_w = resized_image.width();

      const auto top_pad = (dst_max_size - resized_h) / 2;
      const auto left_pad = (dst_max_size - resized_w) / 2;
      const auto bottom_pad = dst_max_size - resized_h - top_pad;
      const auto right_pad = dst_max_size - resized_w - left_pad;

      // Fill for the padding.
      dst.flat_array().fill(PixelTraits<T>::zero());
      // Shift the resized image.
      for (auto y = 0; y < resized_image.height(); ++y)
        for (auto x = 0; x < resized_image.width(); ++x)
          dst(x + left_pad, y + top_pad) = resized_image(x, y);

      const RowVector4i window{
          left_pad, top_pad,                         // top-left corner
          resized_w + left_pad, resized_h + top_pad  // bottom-right corner
      };

      const RowVector4i padding{top_pad, bottom_pad, left_pad, right_pad};

      return std::make_tuple(window, scale, padding);
    }

    template <typename T>
    auto operator()(const ImageView<T>& src, const Vector2i& new_sizes) const
        -> std::tuple<Image<T>, Window, Scale, Padding>
    {
      auto dst = Image<T>{new_sizes};
      auto resize_info = this->operator()(src, dst);
      return std::make_tuple(dst, std::get<0>(resize_info),
                             std::get<1>(resize_info),
                             std::get<2>(resize_info));
    }
  };
  //! @}

} /* namespace Sara */
} /* namespace DO */
