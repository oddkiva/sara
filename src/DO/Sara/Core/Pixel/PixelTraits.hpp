// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_SARA_CORE_PIXEL_PIXELTRAITS_HPP
#define DO_SARA_CORE_PIXEL_PIXELTRAITS_HPP


#include <functional>
#include <stdexcept>

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Pixel/ChannelConversion.hpp>
#include <DO/Sara/Core/Pixel/Pixel.hpp>


namespace DO { namespace Sara {

  template <typename T>
  struct PixelTraits
  {
    typedef T channel_type;

    typedef T pixel_type;

    //! \brief Return zero color value.
    static inline pixel_type zero()
    {
      return 0;
    }

    //! \brief Return min pixel value.
    static inline pixel_type min()
    {
      return channel_min_value<T>();
    }

    //! \brief Return max pixel value.
    static inline pixel_type max()
    {
      return channel_max_value<T>();
    }

    //! \brief Cast functor.
    template <typename U>
    struct Cast
    {
      typedef U pixel_type;

      static inline U apply(T value)
      {
        return static_cast<U>(value);
      }
    };
  };

  template <typename T, int M, int N>
  struct PixelTraits<Matrix<T, M, N>>
  {
    typedef T channel_type;

    typedef Matrix<T, M, N> pixel_type;

    //! \brief Return zero color value.
    static inline pixel_type zero()
    {
      pixel_type zero;
      zero.setZero();
      return zero;
    }

    //! \brief Return min color value.
    static inline pixel_type min()
    {
      pixel_type min;
      min.fill(channel_min_value<T>());
      return min;
    }

    //! \brief Return max color value.
    static inline pixel_type max()
    {
      pixel_type max;
      max.fill(channel_max_value<T>());
      return max;
    }

    //! \brief Cast functor.
    template <typename U>
    struct Cast
    {
      typedef Matrix<U, M, N> pixel_type;

      static inline Matrix<U, M, N> apply(const Matrix<T, M, N>& value)
      {
        return value.template cast<U>();
      }
    };
  };

  template <typename T, typename ColorSpace>
  struct PixelTraits<Pixel<T, ColorSpace>>
  {
    typedef T channel_type;

    typedef Pixel<T, ColorSpace> pixel_type;

    //! \brief Return zero color value.
    static inline pixel_type zero()
    {
      pixel_type zero;
      zero.setZero();
      return zero;
    }

    //! \brief Return min color value.
    static inline pixel_type min()
    {
      pixel_type min;
      min.fill(channel_min_value<T>());
      return min;
    }

    //! \brief Return max color value.
    static inline pixel_type max()
    {
      pixel_type max;
      max.fill(channel_max_value<T>());
      return max;
    }

    //! \brief Cast functor.
    template <typename U>
    struct Cast
    {
      typedef Pixel<U, ColorSpace> pixel_type;

      static inline Pixel<U, ColorSpace> apply(const Pixel<T, ColorSpace>& value)
      {
        return value.template cast<U>();
      }
    };
  };

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_CORE_PIXEL_PIXELTRAITS_HPP */
