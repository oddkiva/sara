#pragma once


#include <DO/Core/StaticAssert.hpp>
#include <DO/Core/EigenExtension.hpp>

#include "colorspace.hpp"
#include "pixel.hpp"


// Color space conversion with floating-point values.
namespace DO {

  //! Convert RGB color to gray color.
  template <typename T>
  inline void rgb_to_gray(const Matrix<T, 3, 1>& rgb, T& gray)
  {
    DO_STATIC_ASSERT(
      !std::numeric_limits<T>::is_integer,
      CONVERSION_FROM_RGB_TO_GRAY_IS_SUPPORTED_ONLY_FOR_FLOATING_POINT_TYPE);

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
    DO_STATIC_ASSERT(
      !std::numeric_limits<T>::is_integer,
      CONVERSION_FROM_GRAY_TO_RGB_IS_SUPPORTED_ONLY_FOR_FLOATING_POINT_TYPE);

    yuv[0] = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2];
    yuv[1] = 0.492*(rgb[2] - yuv[0]);
    yuv[2] = 0.877*(rgb[0] - yuv[0]);
  }

  //! Convert YUV color to RGB color.
  template <typename T>
  inline void yuv_to_rgb(const Matrix<T, 3, 1>& yuv, Matrix<T, 3, 1>& rgb)
  {
    DO_STATIC_ASSERT(
      !std::numeric_limits<T>::is_integer,
      CONVERSION_FROM_GRAY_TO_RGB_IS_SUPPORTED_ONLY_FOR_FLOATING_POINT_TYPE);

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

}

// Unified API.
namespace DO {

  //! Color conversion from RGBA to RGB.
  template <typename T>
  void convert_color(Pixel<T, Rgba>& src, const Pixel<T, Rgb>& dst)
  {
    for (int i = 0; i < 3; ++i)
      dst[i] = src[i];
  }

  //! Pixel conversion from RGB to RGBA.
  template <typename T>
  void convert_color(Pixel<T, Rgb>& src, const Pixel<T, Rgba>& dst)
  {
    using namespace std;
    for (int i = 0; i < 3; ++i)
      dst[i] = src[i];
    dst[3] = numeric_limits<T>::is_integer ? numeric_limits<T>::max() : T(1);
  }
  
  //! \brief Convert color from gray to RGB.
  template <typename T>
  inline void convert_color(T src, Pixel<T, Rgb>& dst)
  {
    gray_to_rgb(src, dst);
  }

  //! \brief Convert color from RGB to YUV.
  template <typename T>
  inline void convert_color(const Pixel<T, Rgb>& src, Pixel<T, Yuv>& dst)
  {
    rgb_to_yuv(src, dst);
  }

  //! \brief Convert color from YUV to RGB.
  template <typename T>
  inline void convert_color(const Pixel<T, Yuv>& src, Pixel<T, Rgb>& dst)
  {
    yuv_to_rgb(src, dst);
  }

  //! \brief Generic color converter to grayscale.
  template <typename ColorSpace> struct ConvertColorToGray;
  
  //! \brief Generic color conversion function to grayscale.
  template <typename T, typename ColorSpace>
  inline void convert_color(const Pixel<T, ColorSpace>& src, T& dst)
  {
    ConvertColorToGray<ColorSpace>::template apply<T>(src, dst);
  }
  
  //! \brief Convert color from RGB to gray.
  template <> struct ConvertColorToGray<Rgb>
  {
    template <typename T>
    static inline void apply(const Matrix<T,3,1>& src, T& dst)
    {
      rgb_to_gray(src, dst);
    }
  };
  
  //! \brief Convert YUV color to gray color.
  template <> struct ConvertColorToGray<Yuv>
  {
    template <typename T>
    static inline void apply(const Matrix<T,3,1>& src, T& dst)
    {
      yuv_to_gray(src, dst);
    }
  };

}