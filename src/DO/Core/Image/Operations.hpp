// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_CORE_IMAGE_OPERATIONS_HPP
#define DO_CORE_IMAGE_OPERATIONS_HPP


#include <DO/Core/Image/Image.hpp>
#include <DO/Core/Pixel/ChannelConversion.hpp>
#include <DO/Core/Pixel/ColorConversion.hpp>
#include <DO/Core/Pixel/PixelTraits.hpp>


// Various utilities for image operations.
namespace DO {

  //! \ingroup Image
  //! @{

  //! \brief Find min and max grayscale values of the image.
  template <typename T, int N>
  inline void find_min_max(T& min, T& max, const Image<T, N>& src)
  {
    const T *src_first = src.data();
    const T *src_last = src_first + src.size();
    min = *std::min_element(src_first, src_last);
    max = *std::max_element(src_first, src_last);
  }

  //! \brief Find min and max pixel values of the image.
  template <typename T, int N, typename Layout>
  void find_min_max(Pixel<T, Layout>& min,
                    Pixel<T, Layout>& max,
                    const Image<Pixel<T, Layout>, N>& src)
  {
    const Pixel<T,Layout> *src_first = src.data();
    const Pixel<T,Layout> *src_last = src_first + src.size();

    min = *src_first;
    max = *src_first;

    for ( ; src_first != src_last; ++src_first)
    {
      min = min.cwiseMin(*src_first);
      max = max.cwiseMax(*src_first);
    }
  }

  //! @}
}


// Generic color conversion of images.
namespace DO {

  //! \ingroup Image
  //! @{

  //! \brief Convert channel type of image.
  template <typename T, typename U, int N>
  void convert_channel(const Image<T, N>& src, Image<U, N>& dst)
  {
    if (dst.sizes() != src.sizes())
      dst.resize(src.sizes());

    const T *src_first = src.data();
    const T *src_last = src_first + src.size();

    U *dst_first = dst.data();

    for ( ; src_first != src_last; ++src_first, ++dst_first)
      convert_channel(*src_first, *dst_first);
  }

  //! \brief Convert color of image.
  template <typename T, typename U, int N>
  void convert_color(const Image<T, N>& src, Image<U, N>& dst)
  {
    if (dst.sizes() != src.sizes())
      dst.resize(src.sizes());

    const T *src_first = src.data();
    const T *src_last = src_first + src.size();

    U *dst_first = dst.data();

    for ( ; src_first != src_last; ++src_first, ++dst_first)
      convert_color(*src_first, *dst_first);
  }

  //! @}
}


// Image rescaling functions
namespace DO {

  //! \ingroup Image
  //! @{

  //! \brief Rescale color values properly for viewing purposes.
  template <typename T, int N>
  inline Image<T, N> color_rescale(const Image<T, N>& src,
                                   const T& a = PixelTraits<T>::min(),
                                   const T& b = PixelTraits<T>::max())
  {
    DO_STATIC_ASSERT(!std::numeric_limits<T>::is_integer,
                     IMPLEMENTATION_NOT_SUPPORTED_FOR_INTEGER_TYPES);

    Image<T, N> dst(src.sizes());

    const T *src_first = src.data();
    const T *src_last = src_first + src.size();
    T *dst_first  = dst.data();

    T min = *std::min_element(src_first, src_last);
    T max = *std::max_element(src_first, src_last);

    if (min == max)
      throw std::runtime_error("Error: cannot rescale image! min == max");

    for ( ; src_first != src_last; ++src_first, ++dst_first)
      *dst_first = a + (b-a)*(*src_first-min)/(max-min);

    return dst;
  }


  //! \brief color rescaling function.
  template <typename T, typename ColorSpace, int N>
  inline Image<Pixel<T, ColorSpace>, N> color_rescale(
    const Image<Pixel<T, ColorSpace>, N>& src,
    const Pixel<T, ColorSpace>& a = PixelTraits<Pixel<T, ColorSpace> >::min(),
    const Pixel<T, ColorSpace>& b = PixelTraits<Pixel<T, ColorSpace> >::max())
  {
    DO_STATIC_ASSERT(!std::numeric_limits<T>::is_integer,
                     IMPLEMENTATION_NOT_SUPPORTED_FOR_INTEGER_TYPES);

    Image<Pixel<T,ColorSpace>, N> dst(src.sizes());

    const Pixel<T, ColorSpace> *src_first = src.data();
    const Pixel<T, ColorSpace> *src_last = src_first + src.size();
    Pixel<T, ColorSpace> *dst_first  = dst.data();

    Pixel<T, ColorSpace> min(*src_first);
    Pixel<T, ColorSpace> max(*src_first);
    for ( ; src_first != src_last; ++src_first)
    {
      min = min.cwiseMin(*src_first);
      max = max.cwiseMax(*src_first);
    }

    if (min == max)
      throw std::runtime_error("Error: cannot rescale image! min == max");

    for (src_first = src.data(); src_first != src_last; 
      ++src_first, ++dst_first)
      *dst_first = a + (*src_first-min).cwiseProduct(b-a).
      cwiseQuotient(max-min);

    return dst;
  }


  //! \brief color rescaling functor helper.
  template <typename T, int N>
  struct ColorRescale
  {
    typedef Image<T, N> ReturnType;
    ColorRescale(const Image<T, N>& src) : src_(src) {}
    ReturnType operator()() const { return color_rescale(src_); }
    const Image<T, N>& src_;
  };

  //! @}
}


#endif /* DO_CORE_IMAGE_OPERATIONS_HPP */