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

#ifndef DO_IMAGEPROCESSING_WARP_HPP
#define DO_IMAGEPROCESSING_WARP_HPP


#include <DO/ImageProcessing/Interpolation.hpp>


namespace DO {
  
  template <typename T, typename S>
  void warp(const Image<T>& src, Image<T>& dst,
            const Matrix<S, 3, 3>& homography_from_dst_to_src,
            const T& default_fill_color = PixelTraits<T>::min())
  {
    typedef typename PixelTraits<T>::template Cast<double>::pixel_type
      DoublePixel;
    typedef typename PixelTraits<T>::channel_type ChannelType;
    typedef Matrix<S, 3, 3> Matrix3;
    typedef Matrix<S, 3, 1> Vector3;
    typedef Matrix<S, 2, 1> Vector2;

    const Matrix3& H = homography_from_dst_to_src;
    
    typename Image<T>::array_iterator it = dst.begin_array();
    for ( ; !it.end(); ++it)
    {
      // Get the corresponding coordinates in the source image.
      Vector3 H_p;
      H_p = H * (Vector3() << it.position().template cast<S>(), 1).finished();
      H_p /= H_p(2);

      // Check if the position is not in the src domain [0,w[ x [0,h[.
      bool position_is_in_src_domain =
        H_p.x() >= 0 || H_p.x() < S(src.width()) ||
        H_p.y() >= 0 || H_p.y() < S(src.height());

      // Fill with either the default value or the interpolated value.
      if (position_is_in_src_domain)
      {
        Vector2 H_p_2(H_p.template head<2>());
        DoublePixel pixel_value( interpolate(src, H_p_2) );
        *it = PixelTraits<DoublePixel>::template Cast<ChannelType>::apply(
          pixel_value);
      }
      else
        *it = default_fill_color;
    }
  }

}


#endif /* DO_IMAGEPROCESSING_WARP_HPP */