#pragma once


#include <DO/Core/Meta.hpp>


// Template meta-programming utilities.
namespace DO { namespace Meta {

  template <int _Value0, int _Value1, int _Value2>
  struct IntArray_3 {
    enum {
      value_0 = _Value0,
      value_1 = _Value1,
      value_2 = _Value2,
      size = 3
    };
  };

  template <typename IntArray, int index> struct Get;

  template <typename IntArray> struct Get<IntArray, 0>
  { enum { value = IntArray::value_0 }; };
  template <typename IntArray> struct Get<IntArray, 1>
  { enum { value = IntArray::value_1 }; };
  template <typename IntArray> struct Get<IntArray, 2>
  { enum { value = IntArray::value_2 }; };
  template <typename IntArray> struct Get<IntArray, 3>
  { enum { value = IntArray::value_3 }; };

} /* namespace Meta */
} /* namespace DO */


// Color spaces and channel names.
namespace DO {

  // ======================================================================== //
  /*!
    \ingroup Core
    \defgroup Color Color
    @{
  
    \defgroup ColorSpace Color spaces
    @{
   */
  // RGB, RGBA color spaces.
  //! Red channel name (RGB, RGBA).
  struct R {};
  //! Green channel name (RGB, RGBA).
  struct G {};
  //! Blue channel name (RGB, RGBA).
  struct B {};
  //! Alpha channel name (RGB, RGBA).
  struct A {};

  // YUV color space.
  // Luminance channel name (YUV).
  struct Y {};
  //! First chrominance name (YUV).
  struct U {};
  //! Second chrominance name (YUV) or Value channel name (HSV).
  struct V {};

  // HSV color space.
  //! Hue channel name (HSV).
  struct H {};
  //! Saturation channel name (HSV).
  struct S {};
  // Value channel name (HSV).
  /*struct V {};*/

  
  // ======================================================================== //
  // Color space and layouts.
  //! Grayscale color space.
  struct Gray {};
  //! RGB color space and layout.
  typedef Meta::Vector3<R,G,B> Rgb;
  //! RGBA color space and layout.
  typedef Meta::Vector4<R,G,B,A> Rgba;
  //! YUV color space and layout.
  typedef Meta::Vector3<Y,U,V> Yuv;
  //! HSV color space and layout.
  typedef Meta::Vector3<H,S,V> Hsv;
  //! @} ColorSpaces

} /* namespace DO */