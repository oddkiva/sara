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

//! @file

#pragma once

#include <DO/Sara/Core/Meta.hpp>
#include <DO/Sara/Core/Pixel/ChannelConversion.hpp>
#include <DO/Sara/Core/Pixel/ColorConversion.hpp>


// Smart color conversion between colorspace regardless of the channel type.
// We will treat the grayscale conversion separately.
namespace DO { namespace Sara {

  namespace Detail {

    //! @{
    //! Generic color conversion functor.
    template <typename SrcColSpace, typename DstColSpace>
    struct SmartConvertColor;

    //! Generic cases.
    template <typename SrcColSpace, typename DstColSpace>
    struct SmartConvertColor
    {
      template <typename T>
      static inline void apply(const Pixel<T, SrcColSpace>& src,
                               Pixel<float, DstColSpace>& dst)
      {
        Pixel<float, SrcColSpace> float_src;
        convert_channel<T, float, SrcColSpace>(src, float_src);
        convert_color(float_src, dst);
      }

      template <typename T>
      static inline void apply(const Pixel<T, SrcColSpace>& src,
                               Pixel<double, DstColSpace>& dst)
      {
        Pixel<double, SrcColSpace> double_src;
        convert_channel<T, double, SrcColSpace>(src, double_src);
        convert_color(double_src, dst);
      }

      template <typename T>
      static inline void apply(const Pixel<float, SrcColSpace>& src,
                               Pixel<T, DstColSpace>& dst)
      {
        Pixel<float, DstColSpace> float_dst;
        convert_color(src, float_dst);
        convert_channel<float, T, DstColSpace>(float_dst, dst);
      }

      template <typename T>
      static inline void apply(const Pixel<double, SrcColSpace>& src,
                               Pixel<T, DstColSpace>& dst)
      {
        Pixel<double, DstColSpace> double_dst;
        convert_color(src, double_dst);
        convert_channel<double, T, DstColSpace>(double_dst, dst);
      }
    };

    //! Particular cases.
    template <typename ColSpace>
    struct SmartConvertColor<ColSpace, ColSpace>
    {
      template <typename T>
      static inline void apply(const Pixel<T, ColSpace>& src,
                               Pixel<double, ColSpace>& dst)
      {
        convert_channel<T, double, ColSpace>(src, dst);
      }

      template <typename T>
      static inline void apply(const Pixel<double, ColSpace>& src,
                               Pixel<T, ColSpace>& dst)
      {
        convert_channel<double, T, ColSpace>(src, dst);
      }

      template <typename T>
      static inline void apply(const Pixel<T, ColSpace>& src,
                               Pixel<float, ColSpace>& dst)
      {
        convert_channel<T, float, ColSpace>(src, dst);
      }

      template <typename T>
      static inline void apply(const Pixel<float, ColSpace>& src,
                               Pixel<T, ColSpace>& dst)
      {
        const Matrix<float, N, 1>& msrc = src;
        const Matrix<T, N, 1>& mdst = dst;
        convert_channel(src, dst);
      }
    };
    //! @}
  }

  //! @{
  //! @brief Smart color conversion from a colorspace to another.
  template <typename ColSpace>
  inline void smart_convert_color(const Pixel<float, ColSpace>& src,
                                  Pixel<double, ColSpace>& dst)
  {
    convert_channel<float, double, ColSpace>(src, dst);
  }

  template <typename ColSpace>
  inline void smart_convert_color(const Pixel<double, ColSpace>& src,
                                  Pixel<float, ColSpace>& dst)
  {
    convert_channel<double, float, ColSpace>(src, dst);
  }

  template <typename SrcColSpace, typename DstColSpace>
  inline void smart_convert_color(const Pixel<float, SrcColSpace>& src,
                                  Pixel<float, DstColSpace>& dst)
  {
    convert_color(src, dst);
  }

  template <typename SrcColSpace, typename DstColSpace>
  inline void smart_convert_color(const Pixel<double, SrcColSpace>& src,
                                  Pixel<double, DstColSpace>& dst)
  {
    convert_color(src, dst);
  }

  template <typename SrcColSpace, typename DstColSpace>
  inline void smart_convert_color(const Pixel<float, SrcColSpace>& src,
                                  Pixel<double, DstColSpace>& dst)
  {
    Pixel<double, SrcColSpace> double_src;
    convert_channel<float, double, SrcColSpace>(src, double_src);
    convert_color(double_src, dst);
  }

  template <typename SrcColSpace, typename DstColSpace>
  inline void smart_convert_color(const Pixel<double, SrcColSpace>& src,
                                  Pixel<float, DstColSpace>& dst)
  {
    Pixel<double, DstColSpace> double_dst;
    convert_color(src, double_dst);
    convert_channel<double, float, DstColSpace>(double_dst, dst);
  }

  template <typename SrcT, typename SrcColSpace, typename DstColSpace>
  inline void smart_convert_color(const Pixel<SrcT, SrcColSpace>& src,
                                  Pixel<float, DstColSpace>& dst)
  {
    Detail::SmartConvertColor<SrcColSpace, DstColSpace>::apply(src, dst);
  }

  template <typename SrcT, typename SrcColSpace, typename DstColSpace>
  inline void smart_convert_color(const Pixel<SrcT, SrcColSpace>& src,
                                  Pixel<double, DstColSpace>& dst)
  {
    Detail::SmartConvertColor<SrcColSpace, DstColSpace>::apply(src, dst);
  }

  template <typename DstT, typename SrcColSpace, typename DstColSpace>
  inline void smart_convert_color(const Pixel<float, SrcColSpace>& src,
                                  Pixel<DstT, DstColSpace>& dst)
  {
    Detail::SmartConvertColor<SrcColSpace, DstColSpace>::apply(src, dst);
  }

  template <typename DstT, typename SrcColSpace, typename DstColSpace>
  inline void smart_convert_color(const Pixel<double, SrcColSpace>& src,
                                  Pixel<DstT, DstColSpace>& dst)
  {
    Detail::SmartConvertColor<SrcColSpace, DstColSpace>::apply(src, dst);
  }

  template <
    typename SrcT, typename DstT, typename SrcColSpace, typename DstColSpace
  >
  inline void smart_convert_color(const Pixel<SrcT, SrcColSpace>& src,
                                  Pixel<DstT, DstColSpace>& dst)
  {
    Pixel<double, SrcColSpace> double_src;
    Pixel<double, DstColSpace> double_dst;
    convert_channel<SrcT, double, SrcColSpace>(src, double_src);
    convert_color(double_src, double_dst);
    convert_channel<double, DstT, DstColSpace>(double_dst, dst);
  }
  //! @}

} /* namespace Sara */
} /* namespace DO */


// Smart color conversion from a colorspace to grayscale regardless of the
// channel type.
namespace DO { namespace Sara {

  //! @brief Convert from 'double' pixel to 'double' grayscale.
  template <typename ColorSpace>
  inline void smart_convert_color(const Pixel<double, ColorSpace>& src,
                                  double& dst)
  {
    convert_color(src, dst);
  }

  //! @brief Convert from 'double' pixel to 'any' grayscale.
  template <typename T, typename ColorSpace>
  inline void smart_convert_color(const Pixel<double, ColorSpace>& src, T& dst)
  {
    double double_dst;
    convert_color(src, double_dst);
    convert_channel(double_dst, dst);
  }

  //! @brief Convert from 'any' pixel to 'double' grayscale.
  template <typename T, typename ColorSpace>
  inline void smart_convert_color(const Pixel<T, ColorSpace>& src, double& dst)
  {
    Pixel<double, ColorSpace> double_src;
    convert_channel<T, double, ColorSpace>(src, double_src);
    convert_color(double_src, dst);
  }

  //! @brief Convert from 'any' pixel to 'any' grayscale.
  template <typename T, typename U, typename ColorSpace>
  inline void smart_convert_color(const Pixel<T, ColorSpace>& src, U& dst)
  {
    Pixel<double, ColorSpace> double_src;
    double double_dst;
    convert_channel<T, double, ColorSpace>(src, double_src);
    convert_color(double_src, double_dst);
    convert_channel(double_dst, dst);
  }

} /* namespace Sara */
} /* namespace DO */


// Smart color conversion from grayscale to another colorspace regardless of
// the channel type.
namespace DO { namespace Sara {

  //! @brief Convert from 'double' grayscale to 'double' pixel.
  template <typename ColorSpace>
  inline void smart_convert_color(double src, Pixel<double, ColorSpace>& dst)
  {
    convert_color(src, dst);
  }

  //! @brief Convert from 'double' grayscale to 'any' pixel.
  template <typename T, typename ColorSpace>
  inline void smart_convert_color(double src, Pixel<T, ColorSpace>& dst)
  {
    Pixel<double, ColorSpace> double_dst;
    convert_color(src, double_dst);
    convert_channel<double, T, ColorSpace>(double_dst, dst);
  }

  //! @brief Convert from 'any' grayscale to 'double' pixel.
  template <typename T, typename ColorSpace>
  inline void smart_convert_color(T src, Pixel<double, ColorSpace>& dst)
  {
    double double_src;
    convert_channel(src, double_src);
    convert_color(double_src, dst);
  }

  //! @brief Convert from 'any' grayscale to 'any' pixel.
  template <typename T, typename U, typename ColorSpace>
  inline void smart_convert_color(T src, Pixel<U, ColorSpace>& dst)
  {
    double double_src;
    Pixel<double, ColorSpace> double_dst;
    convert_channel(src, double_src);
    convert_color(double_src, double_dst);
    convert_channel<double, U, ColorSpace>(double_dst, dst);
  }

} /* namespace Sara */
} /* namespace DO */


// Smart color conversion from 'any' grayscale to 'any' grayscale.
namespace DO { namespace Sara {

  //! @brief Convert from 'any' grayscale to 'any' grayscale.
  template <typename SrcGray, typename DstGray>
  inline void smart_convert_color(SrcGray src, DstGray& dst)
  {
    static_assert(!std::is_same<SrcGray, DstGray>::value,
      "Source and destination grayscale types cannot be identical");
    convert_channel(src, dst);
  }

} /* namespace Sara */
} /* namespace DO */
