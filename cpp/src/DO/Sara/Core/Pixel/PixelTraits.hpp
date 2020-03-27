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

#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Pixel/ChannelConversion.hpp>
#include <DO/Sara/Core/Pixel/Pixel.hpp>

#include <functional>
#include <stdexcept>


namespace DO { namespace Sara {

  //! @addtogroup Color
  //! @{

  template <typename T>
  struct PixelTraits
  {
    using channel_type = T;
    using pixel_type = T;

    static constexpr auto num_channels = 1;

    //! @brief Return zero color value.
    static inline pixel_type zero()
    {
      return 0;
    }

    //! @brief Return min pixel value.
    static inline pixel_type min()
    {
      return channel_min_value<T>();
    }

    //! @brief Return max pixel value.
    static inline pixel_type max()
    {
      return channel_max_value<T>();
    }

    //! @brief Cast functor.
    template <typename U>
    struct Cast
    {
      using pixel_type = U;

      static inline U apply(T value)
      {
        return static_cast<U>(value);
      }
    };
  };

  template <typename T, int M, int N>
  struct PixelTraits<Matrix<T, M, N>>
  {
    using channel_type = T;
    using pixel_type = Matrix<T, M, N>;

    static constexpr auto num_channels = M * N;

    //! @brief Return zero color value.
    static inline pixel_type zero()
    {
      pixel_type zero;
      zero.setZero();
      return zero;
    }

    //! @brief Return min color value.
    static inline pixel_type min()
    {
      pixel_type min;
      min.fill(channel_min_value<T>());
      return min;
    }

    //! @brief Return max color value.
    static inline pixel_type max()
    {
      pixel_type max;
      max.fill(channel_max_value<T>());
      return max;
    }

    //! @brief Cast functor.
    template <typename U>
    struct Cast
    {
      using pixel_type = Matrix<U, M, N>;

      static inline Matrix<U, M, N> apply(const Matrix<T, M, N>& value)
      {
        return value.template cast<U>();
      }
    };
  };

  template <typename T, typename ColorSpace>
  struct PixelTraits<Pixel<T, ColorSpace>>
  {
    using channel_type = T;
    using pixel_type = Pixel<T, ColorSpace>;

    static constexpr auto num_channels = ColorSpace::size;

    //! @brief Return zero color value.
    static inline pixel_type zero()
    {
      pixel_type zero;
      zero.setZero();
      return zero;
    }

    //! @brief Return min color value.
    static inline pixel_type min()
    {
      pixel_type min;
      min.fill(channel_min_value<T>());
      return min;
    }

    //! @brief Return max color value.
    static inline pixel_type max()
    {
      pixel_type max;
      max.fill(channel_max_value<T>());
      return max;
    }

    //! @brief Cast functor.
    template <typename U>
    struct Cast
    {
      using pixel_type = Pixel<U, ColorSpace>;

      static inline Pixel<U, ColorSpace> apply(const Pixel<T, ColorSpace>& value)
      {
        return value.template cast<U>();
      }
    };
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */
