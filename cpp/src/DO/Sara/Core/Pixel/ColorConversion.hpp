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
#include <DO/Sara/Core/Pixel/ColorSpace.hpp>
#include <DO/Sara/Core/Pixel/Pixel.hpp>


// Color space conversion with floating-point values.
namespace DO { namespace Sara {

  //! Convert RGB color to gray color.
  template <typename T, int N>
  inline void rgb_to_gray(const Matrix<T, N, 1>& rgb, T& gray)
  {
    static_assert(
      !std::numeric_limits<T>::is_integer,
      "Conversion from rgb to gray is supported only for floating point type");

    gray = T(0.2125)*rgb[0] + T(0.7154)*rgb[1] + T(0.0721)*rgb[2];
  }

  //! Convert Grayscale color to RGB color.
  template <typename T>
  inline void gray_to_rgb(T gray, Matrix<T, 3, 1>& rgb)
  {
    rgb.fill(gray);
  }

  //! Convert RGB color to YUV color.
  template <typename T>
  inline void rgb_to_yuv(const Matrix<T, 3, 1>& rgb, Matrix<T, 3, 1>& yuv)
  {
    static_assert(
      !std::numeric_limits<T>::is_integer,
      "Conversion from gray to RGB is supported only for floating point type");

    yuv[0] = T(0.299)*rgb[0] + T(0.587)*rgb[1] + T(0.114)*rgb[2];
    yuv[1] = T(0.492)*(rgb[2] - yuv[0]);
    yuv[2] = T(0.877)*(rgb[0] - yuv[0]);
  }

  //! Convert YUV color to RGB color.
  template <typename T>
  inline void yuv_to_rgb(const Matrix<T, 3, 1>& yuv, Matrix<T, 3, 1>& rgb)
  {
    static_assert(
      !std::numeric_limits<T>::is_integer,
      "Conversion from gray to RGB is supported only for floating point type");

    rgb[0] = yuv[0]                     + T(1.13983)*yuv[2];
    rgb[1] = yuv[0] - T(0.39465)*yuv[1] - T(0.58060)*yuv[2];
    rgb[2] = yuv[0] + T(2.03211)*yuv[1];
  }

  //! Convert YUV color to gray color.
  template <typename T>
  inline void yuv_to_gray(const Matrix<T, 3, 1>& yuv, T& gray)
  {
    gray = yuv[0];
  }

  //! Convert gray color to YUV color.
  template <typename T>
  inline void gray_to_yuv(T gray, Matrix<T, 3, 1>& yuv)
  {
    yuv << gray, 0, 0;
  }

} /* namespace Sara */
} /* namspace DO */


// Unified API for color conversion between the source and destination pixels
// with the same channel type.
namespace DO { namespace Sara {

  //! Generic color conversion functor.
  template <typename SrcColSpace, typename DstColSpace> struct ConvertColor;

  //! Pixel conversion from RGBA to RGB.
  template <> struct ConvertColor<Rgba, Rgb>
  {
    template <typename T>
    static inline void apply(const Pixel<T, Rgba>& src, Pixel<T, Rgb>& dst)
    {
      for (int i = 0; i < 3; ++i)
        dst[i] = src[i];
    }
  };

  //! Pixel conversion from RGB to RGBA.
  template <> struct ConvertColor<Rgb, Rgba>
  {
    template <typename T>
    static inline void apply(const Pixel<T, Rgb>& src, Pixel<T, Rgba>& dst)
    {
      using namespace std;
      for (int i = 0; i < 3; ++i)
        dst[i] = src[i];
      dst[3] = numeric_limits<T>::is_integer ? numeric_limits<T>::max() : T(1);
    }
  };

  //! @brief Convert color from RGB to YUV.
  template <> struct ConvertColor<Rgb, Yuv>
  {
    template <typename T>
    static inline void apply(const Pixel<T, Rgb>& src, Pixel<T, Yuv>& dst)
    {
      rgb_to_yuv(src, dst);
    }
  };

  //! @brief Convert color from YUV to RGB.
  template <> struct ConvertColor<Yuv, Rgb>
  {
    template <typename T>
    static inline void apply(const Pixel<T, Yuv>& src, Pixel<T, Rgb>& dst)
    {
      yuv_to_rgb(src, dst);
    }
  };

  //! @brief Convert color from a colorspace to another
  template <typename T, typename SrcColSpace, typename DstColSpace>
  inline void convert_color(const Pixel<T, SrcColSpace>& src,
                            Pixel<T, DstColSpace>& dst)
  {
    ConvertColor<SrcColSpace, DstColSpace>::template apply<T>(src, dst);
  }

  //! @brief Generic color converter to grayscale.
  template <typename ColorSpace> struct ConvertColorToGray;

  //! @brief Generic color conversion function to grayscale.
  template <typename T, typename ColorSpace>
  inline void convert_color(const Pixel<T, ColorSpace>& src, T& dst)
  {
    ConvertColorToGray<ColorSpace>::template apply<T>(src, dst);
  }

  //! @brief Convert color from RGB to gray.
  template <> struct ConvertColorToGray<Rgb>
  {
    template <typename T>
    static inline void apply(const Matrix<T, 3, 1>& src, T& dst)
    {
      rgb_to_gray(src, dst);
    }
  };

  //! @brief Convert color from RGBA to gray.
  template <> struct ConvertColorToGray<Rgba>
  {
    template <typename T>
    static inline void apply(const Matrix<T, 4, 1>& src, T& dst)
    {
      rgb_to_gray(src, dst);
    }
  };

  //! @brief Convert YUV color to gray color.
  template <> struct ConvertColorToGray<Yuv>
  {
    template <typename T>
    static inline void apply(const Matrix<T, 3, 1>& src, T& dst)
    {
      yuv_to_gray(src, dst);
    }
  };

  //! @brief Convert color from gray to RGB.
  template <typename T>
  inline void convert_color(T src, Pixel<T, Rgb>& dst)
  {
    gray_to_rgb(src, dst);
  }

  //! @brief Convert gray color to YUV color.
  template <typename T>
  inline void convert_color(T src, Pixel<T, Yuv>& dst)
  {
    gray_to_yuv<T>(src, dst);
  }

} /* namespace Sara */
} /* namespace DO */
