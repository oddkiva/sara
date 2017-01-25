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

#ifndef DO_SARA_CORE_IMAGE_OPERATIONS_HPP
#define DO_SARA_CORE_IMAGE_OPERATIONS_HPP


#include <DO/Sara/Core/Image/Image.hpp>
#include <DO/Sara/Core/Pixel/SmartColorConversion.hpp>
#include <DO/Sara/Core/Pixel/PixelTraits.hpp>


namespace DO { namespace Sara {

  //! @ingroup Image
  //! @{

  //! @{
  //! @brief Find min and max values of the image.
  template <typename T, int N>
  inline std::pair<T, T> find_min_max(const ImageView<T, N>& src)
  {
    const auto *src_first = src.begin();
    const auto *src_last = src.end();
    return { *std::min_element(src_first, src_last),
             *std::max_element(src_first, src_last) };
  }

  template <typename T, int N, typename ColorSpace>
  std::pair<Pixel<T, ColorSpace>, Pixel<T, ColorSpace>>
  find_min_max(const ImageView<Pixel<T, ColorSpace>, N> &src)
  {
    const auto *src_first = src.begin();
    const auto *src_last = src.end();

    auto min = *src_first;
    auto max = *src_first;

    for ( ; src_first != src_last; ++src_first)
    {
      min = min.cwiseMin(*src_first);
      max = max.cwiseMax(*src_first);
    }

    return { min, max };
  }
  //! @}

  //! @}

} /* namespace Sara */
} /* namespace DO */


// Generic color conversion of images.
namespace DO { namespace Sara {

  //! @ingroup Image
  //! @{

  //! @brief Convert color of image.
  template <typename SrcImageBase, typename DstImageBase>
  void convert(const SrcImageBase& src, DstImageBase& dst)
  {
    if (src.sizes() != dst.sizes())
      throw std::domain_error{
        "Color conversion error: image sizes are not equal!"
      };

    const auto *src_first = src.begin();
    const auto *src_last = src.end();
    auto dst_first = dst.begin();

    for ( ; src_first != src_last; ++src_first, ++dst_first)
      smart_convert_color(*src_first, *dst_first);
  }

  //! @}

} /* namespace Sara */
} /* namespace DO */


// Image rescaling functions
namespace DO { namespace Sara {

  //! @ingroup Image
  //! @{

  //! @{
  //! @brief Rescale color values.
  template <typename T, int N>
  inline void color_rescale(const ImageView<T, N>& src, ImageView<T, N>& dst,
                            const T& a = PixelTraits<T>::min(),
                            const T& b = PixelTraits<T>::max())
  {
    static_assert(
        !std::numeric_limits<T>::is_integer,
        "Color rescaling is not directly applicable on integer types");

    if (src.sizes() != dst.sizes())
      throw std::domain_error{
        "Source and destination image sizes are not equal!"
      };

    auto min = T{};
    auto max = T{};
    std::tie(min, max) = find_min_max(src);

    if (min == max)
    {
      std::cout << "Warning: cannot rescale image! min == max" << std::endl;
      std::cout << "No rescaling will be performed" << std::endl;
      return;
    }

    dst.copy(src);
    dst.cwise_transform_inplace([&a, &b, &min, &max](T& pixel) {
      pixel = a + (b - a) * (pixel - min) / (max - min);
    });
  }

  template <typename T, typename ColorSpace, int N>
  inline void color_rescale(
      const ImageView<Pixel<T, ColorSpace>, N>& src,
      ImageView<Pixel<T, ColorSpace>, N>& dst,
      const Pixel<T, ColorSpace>& a = PixelTraits<Pixel<T, ColorSpace>>::min(),
      const Pixel<T, ColorSpace>& b = PixelTraits<Pixel<T, ColorSpace>>::max())
  {
    static_assert(!std::numeric_limits<T>::is_integer,
                  "Color rescale is not directly applicable on integral types");

    if (src.sizes() != dst.sizes())
      throw std::domain_error{
        "Source and destination image sizes are not equal!"
      };

    using PixelType = Pixel<T, ColorSpace>;
    auto min = PixelType{};
    auto max = PixelType{};
    std::tie(min, max) = find_min_max(src);

    if (min == max)
    {
      std::cout << "Warning: cannot rescale image! min == max" << std::endl;
      std::cout << "No rescaling will be performed" << std::endl;
      return;
    }

    dst.copy(src);
    dst.cwise_transform_inplace([&a, &b, &min, &max](PixelType& pixel) {
      pixel = a + (pixel - min).cwiseProduct(b - a).cwiseQuotient(max - min);
    });
  }

  template <typename T, int N>
  inline Image<T, N> color_rescale(const ImageView<T, N>& in,
                                   const T& a = PixelTraits<T>::min(),
                                   const T& b = PixelTraits<T>::max())
  {
    auto out = Image<T, N>{ in.sizes() };
    color_rescale(in, out, a, b);
    return out;
  }

  struct ColorRescale
  {
    template <typename SrcImageView>
    using OutPixel = typename SrcImageView::pixel_type;

    template <typename Pixel, int N>
    void operator()(const ImageView<Pixel, N>& src, ImageView<Pixel, N>& dst,
                    const Pixel& a = PixelTraits<Pixel>::min(),
                    const Pixel& b = PixelTraits<Pixel>::max()) const
    {
      color_rescale(src, dst, a, b);
    }
  };
  //! @}

  //! @}


  //! @{
  //! @brief Get the sub-image of an image.
  template <typename T, int N>
  Image<T, N> crop(const ImageView<T, N>& src,
                   const typename ImageView<T, N>::vector_type& begin_coords,
                   const typename ImageView<T, N>::vector_type& end_coords)
  {
    auto dst = Image<T, N>{ end_coords - begin_coords };

    auto src_it = src.begin_subarray(begin_coords, end_coords);
    for (auto dst_it = dst.begin() ; dst_it != dst.end(); ++dst_it, ++src_it)
    {
      // If a and b are coordinates out bounds.
      if (src_it.position().minCoeff() < 0 ||
          (src_it.position() - src.sizes()).minCoeff() >= 0)
        *dst_it = PixelTraits<T>::min();
      else
        *dst_it = *src_it;
    }

    return dst;
  }

  template <typename T>
  inline Image<T> crop(const ImageView<T>& src, int top_left_x, int top_left_y,
                       int width, int height)
  {
    Vector2i begin_coords{ top_left_x, top_left_y };
    Vector2i end_coords{ top_left_x + width, top_left_y + height };
    return get_subimage(src, begin_coords, end_coords);
  }

  template <typename T>
  inline Image<T> crop(const ImageView<T>& src, int center_x, int center_y,
                       int radius)
  {
    return get_subimage(src, center_x - radius, center_y - radius,
                        2 * radius + 1, 2 * radius + 1);
  }
  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_CORE_IMAGE_OPERATIONS_HPP */
