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
  inline void find_min_max(T& min, T& max, const Image<T, N>& src)
  {
    const auto *src_first = src.begin();
    const auto *src_last = src.end();
    min = *std::min_element(src_first, src_last);
    max = *std::max_element(src_first, src_last);
  }

  template <typename T, int N, typename ColorSpace>
  void find_min_max(Pixel<T, ColorSpace>& min,
                    Pixel<T, ColorSpace>& max,
                    const Image<Pixel<T, ColorSpace>, N>& src)
  {
    const auto *src_first = src.begin();
    const auto *src_last = src.end();

    min = *src_first;
    max = *src_first;

    for ( ; src_first != src_last; ++src_first)
    {
      min = min.cwiseMin(*src_first);
      max = max.cwiseMax(*src_first);
    }
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
  template <typename T, typename U, int N>
  void convert(const Image<T, N>& src, Image<U, N>& dst)
  {
    dst.resize(src.sizes());

    const auto *src_first = src.data();
    const auto *src_last = src.end();

    U *dst_first = dst.data();

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
  inline Image<T, N> color_rescale(const Image<T, N>& src,
                                   const T& a = PixelTraits<T>::min(),
                                   const T& b = PixelTraits<T>::max())
  {
    static_assert(
      !std::numeric_limits<T>::is_integer,
      "Color rescaling is not directly applicable on integer types");

    const auto *src_first = src.begin();
    const auto *src_last = src.end();

    auto min = *std::min_element(src_first, src_last);
    auto max = *std::max_element(src_first, src_last);

    if (min == max)
    {
      std::cout << "Warning: cannot rescale image! min == max" << std::endl;
      std::cout << "No rescaling will be performed" << std::endl;
      return src;
    }

    Image<T, N> dst{ src.sizes() };
    T *dst_first = dst.begin();
    for ( ; src_first != src_last; ++src_first, ++dst_first)
      *dst_first = a + (b-a)*(*src_first-min)/(max-min);

    return dst;
  }

  template <typename T, typename ColorSpace, int N>
  inline Image<Pixel<T, ColorSpace>, N> color_rescale(
    const Image<Pixel<T, ColorSpace>, N>& src,
    const Pixel<T, ColorSpace>& a = PixelTraits<Pixel<T, ColorSpace> >::min(),
    const Pixel<T, ColorSpace>& b = PixelTraits<Pixel<T, ColorSpace> >::max())
  {
    static_assert(
      !std::numeric_limits<T>::is_integer,
      "Color rescale is not directly applicable on integral types");

    Image<Pixel<T, ColorSpace>, N> dst{ src.sizes() };

    const auto *src_first = src.data();
    const auto *src_last = src_first + src.size();
    auto *dst_first  = dst.data();

    auto min = *src_first;
    auto max = *src_first;
    for ( ; src_first != src_last; ++src_first)
    {
      min = min.cwiseMin(*src_first);
      max = max.cwiseMax(*src_first);
    }

    if (min == max)
    {
      std::cout << "Warning: cannot rescale image! min == max" << std::endl;
      std::cout << "No rescaling will be performed" << std::endl;
      return src;
    }

    for (src_first = src.data(); src_first != src_last;
      ++src_first, ++dst_first)
      *dst_first = a + (*src_first-min).cwiseProduct(b-a).
      cwiseQuotient(max-min);

    return dst;
  }

  struct ColorRescale
  {
    template <typename Image>
    using ReturnType = Image;

    template <typename Image>
    ReturnType<Image> operator()(const Image& src) const
    {
      return color_rescale(src);
    }
 };
  //! @}

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_CORE_IMAGE_OPERATIONS_HPP */
