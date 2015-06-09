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

#ifndef DO_SARA_CORE_PIXEL_TYPEDEFS_HPP
#define DO_SARA_CORE_PIXEL_TYPEDEFS_HPP


#include <DO/Sara/Core/Pixel/ChannelConversion.hpp>
#include <DO/Sara/Core/Pixel/ColorSpace.hpp>
#include <DO/Sara/Core/Pixel/Pixel.hpp>


namespace DO { namespace Sara {

  //! \defgroup ColorTypes Color typedefs
  //! @{

  // ======================================================================== //
  //! Macro for generic color typedefs like in OpenGL.
#define DEFINE_GENERIC_COLOR_TYPEDEFS(N)            \
  /*! \brief Color{NumChannels}{ChannelType} */     \
  typedef Matrix<unsigned char, N, 1> Color##N##ub; \
  /*! \brief Color{NumChannels}{ChannelType} */     \
  typedef Matrix<char, N, 1> Color##N##b;           \
  /*! \brief Color{NumChannels}{ChannelType} */     \
  typedef Matrix<unsigned short, N, 1> Color##N##us;\
  /*! \brief Color{NumChannels}{ChannelType} */     \
  typedef Matrix<short, N, 1> Color##N##s;          \
  /*! \brief Color{NumChannels}{ChannelType}. */    \
  typedef Matrix<unsigned int, N, 1> Color##N##ui;  \
  /*! \brief Color{NumChannels}{ChannelType} */     \
  typedef Matrix<int, N, 1> Color##N##i;            \
  /*! \brief Color{NumChannels}{ChannelType} */     \
  typedef Matrix<float, N, 1> Color##N##f;          \
  /*! \brief Color{NumChannels}{ChannelType} */     \
  typedef Matrix<double, N, 1> Color##N##d;

  DEFINE_GENERIC_COLOR_TYPEDEFS(3)
  DEFINE_GENERIC_COLOR_TYPEDEFS(4)
#undef DEFINE_GENERIC_COLOR_TYPEDEFS

  // ======================================================================== //
  //! Macro for color typedefs.
#define DEFINE_COLOR_TYPES(colorspace)                      \
  /*! \brief {ColorSpace}{BitDepthPerChannel} */            \
  typedef Pixel<unsigned char, colorspace> colorspace##8;   \
  /*! \brief {PixelSpace}{BitDepthPerChannel} */            \
  typedef Pixel<unsigned short, colorspace> colorspace##16; \
  /*! \brief {PixelSpace}{BitDepthPerChannel} */            \
  typedef Pixel<unsigned int, colorspace> colorspace##32;   \
  /*! \brief {PixelSpace}{BitDepthPerChannel} */            \
  typedef Pixel<char, colorspace> colorspace##8s;           \
  /*! \brief {PixelSpace}{BitDepthPerChannel} */            \
  typedef Pixel<short, colorspace> colorspace##16s;         \
  /*! \brief {PixelSpace}{BitDepthPerChannel} */            \
  typedef Pixel<int, colorspace> colorspace##32s;           \
  /*! \brief {PixelSpace}{BitDepthPerChannel} */            \
  typedef Pixel<float, colorspace> colorspace##32f;         \
  /*! \brief {PixelSpace}{BitDepthPerChannel} */            \
  typedef Pixel<double, colorspace> colorspace##64f;

  DEFINE_COLOR_TYPES(Rgb)
  DEFINE_COLOR_TYPES(Rgba)
  DEFINE_COLOR_TYPES(Yuv)
#undef DEFINE_COLOR_TYPES

  //! @} ColorTypes


  // ======================================================================== //
  //! \defgroup PrimaryColors Primary Colors in RGB.
  //! @{

  //! White color function.
  template <typename T> inline Matrix<T, 3, 1> white()
  {
    return Matrix<T,3,1>(
      channel_max_value<T>(),
      channel_max_value<T>(),
      channel_max_value<T>() );
  }
  //! Black color function.
  template <typename T> inline Matrix<T, 3, 1> black()
  {
    return Matrix<T,3,1>(
      channel_min_value<T>(),
      channel_min_value<T>(),
      channel_min_value<T>() );
  }
  //! Red color function.
  template <typename T> inline Matrix<T, 3, 1> red()
  {
    return Matrix<T,3,1>(
      channel_max_value<T>(),
      channel_min_value<T>(),
      channel_min_value<T>() );
  }
  //! Green color function.
  template <typename T> inline Matrix<T, 3, 1> green()
  {
    return Matrix<T,3,1>(
      channel_min_value<T>(),
      channel_max_value<T>(),
      channel_min_value<T>() );
  }
  //! Blue color function.
  template <typename T> inline Matrix<T, 3, 1> blue()
  {
    return Matrix<T,3,1>(
      channel_min_value<T>(),
      channel_min_value<T>(),
      channel_max_value<T>() );
  }
  //! Cyan color function.
  template <typename T> inline Matrix<T, 3, 1> cyan()
  {
    return Matrix<T,3,1>(
      channel_min_value<T>(),
      channel_max_value<T>(),
      channel_max_value<T>() );
  }
  //! Yellow color function.
  template <typename T> inline Matrix<T, 3, 1> yellow()
  {
    return Matrix<T,3,1>(
      channel_max_value<T>(),
      channel_max_value<T>(),
      channel_min_value<T>() );
  }
  //! Magenta color function.
  template <typename T> inline Matrix<T, 3, 1> magenta()
  {
    return Matrix<T,3,1>(
      channel_max_value<T>(),
      channel_min_value<T>(),
      channel_max_value<T>() );
  }

  //! Primary color definition.
#define DEFINE_COLOR_CONSTANT(Name, function)       \
  /*! \brief Return primary color of type Rgb8. */  \
  const Rgb8 Name##8(function<unsigned char>());    \
  /*! \brief Return primary color of type Rgb8s. */ \
  const Rgb8s Name##8s(function<char>());           \
  /*! \brief Return primary color of type Rgb16. */ \
  const Rgb16 Name##16(function<unsigned short>()); \
  /*! \brief Return primary color of type Rgb16s. */\
  const Rgb16s Name##16s(function<short>());        \
  /*! \brief Return primary color of type Rgb32. */ \
  const Rgb32 Name##32(function<unsigned int>());   \
  /*! \brief Return primary color of type Rgb32s. */\
  const Rgb32s Name##32s(function<int>());          \
  /*! \brief Return primary color of type Rgb32f. */\
  const Rgb32f Name##32f(function<float>());        \
  /*! \brief Return primary color of type Rgb64f. */\
  const Rgb64f Name##64f(function<double>());

  DEFINE_COLOR_CONSTANT(Red, red)
  DEFINE_COLOR_CONSTANT(Green, green)
  DEFINE_COLOR_CONSTANT(Blue, blue)
  DEFINE_COLOR_CONSTANT(Cyan, cyan)
  DEFINE_COLOR_CONSTANT(Magenta, magenta)
  DEFINE_COLOR_CONSTANT(Yellow, yellow)
  DEFINE_COLOR_CONSTANT(Black, black)
  DEFINE_COLOR_CONSTANT(White, white)
#undef DEFINE_COLOR_CONSTANT

  //! @}


} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_CORE_PIXEL_TYPEDEFS_HPP */
