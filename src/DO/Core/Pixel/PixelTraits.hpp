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

#ifndef DO_CORE_PIXEL_PIXELTRAITS_HPP
#define DO_CORE_PIXEL_PIXELTRAITS_HPP


#include <functional>
#include <stdexcept>

#include <DO/Core/EigenExtension.hpp>
#include <DO/Core/Pixel/Pixel.hpp>


namespace DO {

  template <typename T>
  struct PixelTraits
  {
    typedef T channel_type;

    typedef T pixel_type;

    template <typename U>
    struct Cast { typedef U pixel_type; };
  };

  template <typename T, int M, int N>
  struct PixelTraits<Matrix<T, M, N> >
  {
    typedef T channel_type;

    typedef Matrix<T, M, N> pixel_type;

    template <typename U>
    struct Cast { typedef Matrix<T, M, N> pixel_type; };
  };

  template <typename T, typename ColorSpace>
  struct PixelTraits<Pixel<T, ColorSpace> >
  {
    typedef T channel_type;

    typedef Pixel<T, ColorSpace> pixel_type;

    template <typename U>
    struct Cast { typedef Pixel<U, ColorSpace> pixel_type; };
  };

}


#endif /* DO_CORE_PIXEL_PIXELTRAITS_HPP */