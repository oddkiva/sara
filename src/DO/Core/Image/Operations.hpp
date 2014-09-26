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


// Various utilities for image operations.
namespace DO {

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

  //! \brief Return min color value.
  template <typename T>
  inline T color_min_value()
  {
    return channel_min_value<T>();
  }

  //! \brief Return max color value.
  template <typename T>
  inline T color_max_value()
  {
    return channel_max_value<T>();
  }

  //! \brief Return min color value.
  template <typename T, int N>
  inline Matrix<T, N, 1> color_min_value()
  {
    Matrix<T, N, 1> min;
    min.fill(channel_min_value<T>());
    return min;
  }

  //! \brief Return max color value.
  template <typename T, int N>
  inline Matrix<T, N, 1> color_max_value()
  {
    Matrix<T, N, 1> min;
    min.fill(channel_max_value<T>());
    return min;
  }

}

// Generic color conversion of images.
namespace DO {

  //! \brief Generic image converter class.
  template <typename T, typename U, int N>
  struct ConvertImage
  {
    //! \brief Implementation of the image conversion.
    static void apply(Image<T, N>& dst, const Image<U, N>& src)
    {
      if (dst.sizes() != src.sizes())
        dst.resize(src.sizes());

      const U *src_first = src.data();
      const U *src_last = src_first + src.size();

      T *dst_first  = dst.data();

      for ( ; src_first != src_last; ++src_first, ++dst_first)
        convert_color(*dst_first, *src_first);
    }
  };

  //! \brief Specialized image converter class.
  template <typename T, int N>
  struct ConvertImage<T, T, N>
  {
    //! \brief Implementation of the image conversion.
    static void apply(Image<T, N>& dst, const Image<T, N>& src)
    {
      dst = src;
    }
  };


  //! \brief Unified API for the image conversion.
  template <typename T, typename U, int N>
  inline void convert(Image<T, N>& dst, const Image<U, N>& src)
  {
    ConvertImage<T,U,N>::apply(dst, src);
  }

}

// Image rescaling functions
namespace DO {
  //! \brief Rescales color values properly for viewing purposes.
  template <typename T, int N>
  inline Image<T, N> color_rescale(const Image<T, N>& src,
                                   const T& a = color_min_value<T>(),
                                   const T& b = color_max_value<T>())
  {
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
  template <typename T, typename Layout, int N>
  inline Image<Pixel<T,Layout>, N>
  color_rescale(const Image<Pixel<T,Layout>, N>& src,
                const Matrix<T, Layout::size, 1>& a = color_min_value<T, N>(),
                const Matrix<T, Layout::size, 1>& b = color_max_value<T, N>())
  {
    Image<Pixel<T,Layout>, N> dst(src.sizes());

    const Pixel<T,Layout> *src_first = src.data();
    const Pixel<T,Layout> *src_last = src_first + src.size();
    Pixel<T,Layout> *dst_first  = dst.data();

    Pixel<T,Layout> min(*src_first);
    Pixel<T,Layout> max(*src_first);
    for ( ; src_first != src_last; ++src_first)
    {
      min = min.cwiseMin(*src_first);
      max = max.cwiseMax(*src_first);
    }

    if (min == max)
    {
      std::cerr << "Warning: min == max!" << std::endl;
      return dst;
    }

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

}


#endif /* DO_CORE_IMAGE_OPERATIONS_HPP */