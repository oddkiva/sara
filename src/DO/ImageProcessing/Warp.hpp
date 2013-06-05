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

namespace DO {
  
  template <typename T, typename S>
  bool warp(Image<T>& dst, const Image<T>& src,
            const Matrix<S, 3, 3>& homographyFromPatchToImg,
            const T& defaultFillColor = ColorTraits<T>::min(),
            bool stopIfOutOfRange = false)
  {
    typedef Matrix<S, 3, 3> Matrix3;
    typedef Matrix<S, 3, 1> Vector3;
    typedef Matrix<S, 2, 1> Vector2;

    typename Image<T>::range_iterator dst_it = dst.begin_range();
    typename Image<T>::range_iterator dst_end = dst.end_range();
    const Matrix3& H = homographyFromPatchToImg;
    
    bool isInsideSourceImage = true;

    for ( ; dst_it != dst_end; ++dst_it)
    {
      // Get the corresponding coordinates in the source image.
      Vector3 H_p;
      H_p = H * (Vector3() << dst_it.coords().template cast<S>(), S(1)).finished();
      H_p /= H_p(2);
      // Check if the position is not in the image domain [0,w[ x [0,h[.
      bool posNotInImageDomain =
        H_p.x() < S(0) || H_p.x() >= S(src.width()-1) ||
        H_p.y() < S(0) || H_p.y() >= S(src.height()-1);
      if (posNotInImageDomain && stopIfOutOfRange)
        return false;
      if (posNotInImageDomain && isInsideSourceImage)
        isInsideSourceImage = false;
      // Fill with either the default value or the interpolated value.
      if (posNotInImageDomain)
        *dst_it = defaultFillColor;
      else
        convertColor(*dst_it, interpolate(src, H_p.template head<2>().eval()));
    }

    return isInsideSourceImage;
  }

}

#endif /* DO_IMAGEPROCESSING_WARP_HPP */
