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

#ifndef DO_CORE_IMAGE_SUBIMAGE_HPP
#define DO_CORE_IMAGE_SUBIMAGE_HPP


#include <DO/Core/Image/Image.hpp>
#include <DO/Core/Pixel/ChannelConversion.hpp>


namespace DO {

  //! Get the subimage of an image.
  template <typename T, int N>
  Image<T, N> subimage(const Image<T, N>& src,
                       const Matrix<int, N, 1>& begin_coords,
                       const Matrix<int, N, 1>& end_coords)
  {
    Image<T,N> dst(begin_coords - end_coords);

    typedef typename Image<T, N>::const_subarray_iterator const_subarray_iterator;
    const_subarray_iterator src_it = src.begin_subrange(a, b);

    for (typename Image<T, N>::iterator dst_it = dst.begin();
         dst_it != dst.end(); ++dst_it, ++src_it)
    {
      // If a and b are coordinates out bounds.
      if ((src_it.position() - begin_coords).minCoeff() < 0 ||
          (src_it.position() - end_coords).minCoeff() >= 0)
        *dst_it = color_min_value<T>();
      else
        *dst_it = *src_it;
    }
    return dst;
  }

  //! \brief Get the subimage of an image.
  template <typename T>
  inline Image<T> subimage(const Image<T>& src, int top_left_x, int top_left_y,
                    int width, int height)
  {
    Vector2i begin_coords(top_left_x, top_left_y);
    Vector2i end_coords(top_left_x+width, top_left_y+height);
    return subimage(src, begin_coords, end_coords);
  }

  //! \brief Get the subimage of an image.
  template <typename T>
  inline Image<T> subimage(const Image<T>& src, int center_x, int center_y,
                           int radius)
  {
    return subimage(src, center_x-r, center_y-r, 2*r+1, 2*r+1);
  }

} /* namespace DO */



#endif /* DO_CORE_IMAGE_SUBIMAGE_HPP */