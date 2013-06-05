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

#ifndef DO_CORE_IMAGEPATCH_HPP
#define DO_CORE_IMAGEPATCH_HPP

namespace DO {

  // \todo: test.
  template <typename T, int N>
  Image<T, N> getImagePatch(const Image<T, N>& src, 
                            const Matrix<int, N, 1>& a,
                            const Matrix<int, N, 1>& b)
  {
    Image<T,N> dst(b-a);
    dst.array().fill(ColorTraits<T>::zero());
    CoordsIterator<N> c(a,b), end;
    for (typename Image<T, N>::iterator dst_it = dst.begin();
         dst_it != dst.end(); ++dst_it, ++c)
    {
      if ((*c-a).minCoeff() < 0 || (b-*c).minCoeff() >= 0)
        continue;
      *dst_it = src(*++c);
    }
    return dst;
  }

  // No big fuss, it just works.
  template <typename T>
  Image<T> getImagePatch(const Image<T>& src, int x, int y, int w, int h)
  {
    Image<T> patch(w,h);
    patch.array().fill(ColorTraits<T>::zero());
    for (int v = 0; v < h; ++v)
    {
      if (y+v < 0 || y+v > src.height()-1)
        continue;
      for (int u = 0; u < w; ++u)
      {
        if ( x+u < 0 || x+u > src.width()-1 )
          continue;
        patch(u,v) = src(x+u,y+v);
      }
    }
    return patch;
  }

  // Helper
  template <typename T>
  inline Image<T> getImagePatch(const Image<T>& src, int x, int y, int r)
  {
    return getImagePatch(src, x-r, y-r, 2*r+1, 2*r+1);
  }

} /* namespace DO */


#endif /* DO_CORE_IMAGEPATCH_HPP */