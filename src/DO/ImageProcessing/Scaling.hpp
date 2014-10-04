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

//! @file

#ifndef DO_IMAGEPROCESSING_SCALING_HPP
#define DO_IMAGEPROCESSING_SCALING_HPP


#include <DO/Core/Image/Image.hpp>
#include <DO/Core/Image/Operations.hpp>
#include <DO/ImageProcessing/Deriche.hpp>
#include <DO/ImageProcessing/Interpolation.hpp>


namespace DO {

  /*!
    \ingroup ImageProcessing
    \defgroup Scaling Reduce, enlarge, warp image...
    @{
   */

  //! \brief Upscale image.
  template <typename T, int N>
  Image<T, N> upscale(const Image<T, N>& src, int fact)
  {
    Image<T, N> dst(src.sizes()*fact);
    typename Image<T, N>::array_iterator it(dst.begin_array());
    for ( ; !it.end(); ++it)
      *it = src(it.position() / fact);
    return dst;
  }

  //! \brief Downscale image.
  template <typename T, int N>
  Image<T, N> downscale(const Image<T, N>& src, int fact)
  {
    Image<T, N> dst(src.sizes()/fact);
    typename Image<T, N>::array_iterator it(dst.begin_array());
    for ( ; !it.end(); ++it)
      *it = src(it.position()*fact);
    return dst;
  }

  //! \brief Find min and max coefficient of a vector.
  template <typename T, int N>
  inline std::pair<T, T> range(const Matrix<T, N, 1>& v)
  {
    return std::make_pair(
      *std::min_element(v.data(), v.data()+N),
      *std::max_element(v.data(), v.data()+N)
    );
  }

  //! \brief Reduce image.
  template <typename T, int N>
  Image<T, N> reduce(const Image<T, N>& src, Matrix<int, N, 1> new_sizes,
                     bool keep_ratio=false)  
  {
    // Typedefs.
    typedef typename PixelTraits<T>::template Cast<double>::pixel_type
      DoublePixel;
    typedef typename PixelTraits<T>::channel_type ChannelType;

    // Convert in floating type scalar.
    Image<DoublePixel, N> double_src;
    double_src = src.convert<DoublePixel>();

    Matrix<double, N, 1> original_sizes;
    Matrix<double, N, 1> scale_factors;
    Matrix<double, N, 1> sigmas;
    std::pair<double, double> min_max;

    // Determine the right blurring factor.
    original_sizes = src.sizes().template cast<double>();
    scale_factors =
      original_sizes.cwiseQuotient(new_sizes.template cast<double>());
    min_max = range(scale_factors);
    if (keep_ratio)
    {
      scale_factors.fill(min_max.second);
      new_sizes = (original_sizes/min_max.second).template cast<int>();
    }
    sigmas = 1.5*((scale_factors.array().sqrt()-.99).matrix());

    // Blur with Deriche filter.
    inplace_deriche_blur(double_src, sigmas);

    // Create the new image by interpolating pixel values.
    Image<T, N> dst(new_sizes);
    typename Image<T, N>::array_iterator dst_it(dst.begin_array());
    for ( ; !dst_it.end(); ++dst_it)
    {
      Matrix<double, N, 1> position( dst_it.position()
        .template cast<double>()
        .cwiseProduct(scale_factors) );

      DoublePixel double_pixel_value(interpolate(double_src, position));
      *dst_it = PixelTraits<DoublePixel>::template Cast<ChannelType>::apply(
        double_pixel_value);
    }

    return dst;
  }
  
  //! \brief Reduce image.
  template <typename T>
  inline  Image<T, 2> reduce(const Image<T, 2>& image, int w, int h,
                             bool keep_ratio = false)
  {
    return reduce(image, Vector2i(w,h), keep_ratio);
  }
  
  //! \brief Reduce image.
  template <typename T>
  inline Image<T, 3> reduce(const Image<T, 3>& image, int w, int h, int d,
                            bool keep_ratio = false)
  {
    return reduce(image, Vector3i(w,h,d), keep_ratio);
  }
  
  //! \brief Reduce image.
  template <typename T,int N>
  inline Image<T, N> reduce(const Image<T, N>& image, double fact)  
  {
    Matrix<double, N, 1> new_sizes;
    new_sizes = image.sizes().template cast<double>() / fact;
    return reduce(image, new_sizes.template cast<int>().eval());
  }

  //! \brief Reduce image.
  template <typename T, int N>
  inline Image<T, N> enlarge(const Image<T, N>& image,
                             Matrix<int, N, 1> new_sizes,
                             bool keep_ratio = false)
  {
    // Typedefs.
    typedef typename PixelTraits<T>::template Cast<double>::pixel_type
      DoublePixel;
    typedef typename PixelTraits<T>::channel_type ChannelType;
    typedef Matrix<double, N, 1> DoubleCoords;

    // Determine the right blurring factor.
    DoubleCoords original_sizes(image.sizes().template cast<double>());
    DoubleCoords scale_factor = original_sizes.cwiseQuotient(
      new_sizes.template cast<double>());
    std::pair<double, double> min_max = range(scale_factor);

    if (keep_ratio)
    {
      scale_factor.fill(min_max.second);
      new_sizes = (original_sizes/min_max.second).template cast<int>();
    }

    // Create the new image by interpolation.
    Image<T, N> dst(new_sizes);
    typename Image<T, N>::array_iterator dst_it(dst.begin_array());
    for ( ; !dst_it.end(); ++dst_it)
    {
      DoubleCoords position;
      position = dst_it.position()
        .template cast<double>()
        .cwiseProduct(scale_factor);
      
      DoublePixel double_pixel_value(interpolate(image, position));
      *dst_it = PixelTraits<DoublePixel>::template Cast<ChannelType>::apply(
        double_pixel_value);
    }
    return dst;
  }

  //! \brief Enlarge image.
  template <typename T>
  inline Image<T, 2> enlarge(const Image<T, 2>& image, int w, int h,
                             bool keep_ratio = false)
  {
    return enlarge(image, Point2i(w,h), keep_ratio);
  }

  //! \brief Enlarge image.
  template <typename T>
  inline Image<T, 3> enlarge(const Image<T, 3>& image, int w, int h, int d,
                             bool keep_ratio = false)
  {
    return enlarge(image, Vector3i(w,h,d), keep_ratio);
  }

  //! \brief Enlarge image.
  template <typename T,int N>
  inline Image<T, N> enlarge(const Image<T, N>& image, double fact)  
  {
    Matrix<double, N, 1> new_sizes;
    new_sizes = image.sizes().template cast<double>()*fact;
    return enlarge(image, new_sizes.template cast<int>().eval());
  }

  //! @} file
}


#endif /* DO_IMAGEPROCESSING_SCALING_HPP */