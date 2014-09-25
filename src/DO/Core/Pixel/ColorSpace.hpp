#ifndef DO_CORE_PIXEL_COLORSPACE_HPP
#define DO_CORE_PIXEL_COLORSPACE_HPP


#include <DO/Core/Meta.hpp>


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


#endif /* DO_CORE_PIXEL_COLORSPACE_HPP */