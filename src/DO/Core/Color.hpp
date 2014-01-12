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

#ifndef DO_CORE_COLOR_HPP
#define DO_CORE_COLOR_HPP

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

  // CMYK color space.
  //! Cyan channel name (CMYK).
  struct C {};
  //! Magenta channel name (CMYK).
  struct M {};
  //! Yellow channel name (CMYK) or Luminance channel name (YUV).
  struct Y {};
  //! Black channel name (CMYK).
  struct K {};

  // YUV color space.
  // Luminance channel name (YUV).
  /*struct Y {};*/
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
  //! RGBA color space and layout.
  typedef Meta::Vector4<R,G,B,A> Rgba;
  //! ARGB color space and layout.
  typedef Meta::Vector4<A,R,G,B> Argb;
  //! ABGR color space and layout.
  typedef Meta::Vector4<A,B,G,R> Abgr;
  //! BGRA color space and layout.
  typedef Meta::Vector4<B,R,G,A> Bgra;
  //! CMYK color space and layout.
  typedef Meta::Vector4<C,M,Y,K> Cmyk;
  //! RGB color space and layout.
  typedef Meta::Vector3<R,G,B> Rgb;
  //! BGR color space and layout.
  typedef Meta::Vector3<B,G,R> Bgr;
  //! YUV color space and layout.
  typedef Meta::Vector3<Y,U,V> Yuv;
  //! HSV color space and layout.
  typedef Meta::Vector3<H,S,V> Hsv;
  //! @} ColorSpaces

  
  // ======================================================================== //
  /*!
    \brief Lightweight template color class with some flexibility.

    The color space and layout can have some flexibility.  
    Here are some supported color space and layout examples:
    RGB, BGR, RGBA, BGRA, ARGB, YUV, HSV.

    However every channel *must* have the same bit depth. For complex packed 
    pixels with complex color layouts, consider Boost.GIL instead.

    Also note this may not be intended for video processing!

    Anyways, I wanted to do some image processing as conveniently as possible
    because:
    - I want to avoid introducing a dependency with Boost in the Core module
    because it is too big...
    - I had a bad user experience with the API of Boost.GIL. It seems neither 
    easy to remember nor easy to use and I wasted a lot of time with the 
    documentation.
   */
  template <typename T, typename Layout = Rgb>
  class Color : public Matrix<T, Layout::size, 1>
  {
    typedef Matrix<T, Layout::size, 1> Base;

  public:
    //! Channel type.
    typedef T ChannelType;
    //! Layout type (RGB, BGR, RGBA, CMYK, ...)
    typedef Layout ColorLayout;
    //! Default constructor.
    inline Color() : Base() {}
    //! Custom constructor.
    inline Color(T x, T y, T z) : Base(x, y, z) {}
    //! Custom constructor.
    inline Color(T x, T y, T z, T t) : Base(x, y, z, t) {}
    //! Copy constructor.
    inline Color(const Base& x) : Base(x) {}
    //! Assignment operator.
    template <typename OtherDerived>
    inline Color& operator=(const Eigen::MatrixBase<OtherDerived>& other)
    { this->Base::operator=(other); return *this; }
    //! Constant channel accessor.
    template <typename Channel>
    inline const T& channel() const
    { return (*this)[Meta::IndexOf<Layout, Channel>::value]; }
    //! Mutable channel accessor.
    template <typename Channel>
    inline ChannelType& channel()
    { return (*this)[Meta::IndexOf<Layout, Channel>::value]; }
    //! Returns the number of channels.
    int numChannels() const
    { return Layout::size; }
  };

  // ======================================================================== //
  /*!
    \defgroup ChannelAccessors Short mutable and non-mutable channel accessors

    Example: 
    <code>red(c)</code> is equivalent to <code>c.channel<R>()</code>
    
    It is shorter and may be less error prone if we forget the color layout.
    @{
   */

  //! \brief Macro that defines short mutable and non-mutable channel accessors.
#define DEFINE_COLOR_CHANNEL_GETTERS(channelName, channelTag) \
  /*! \brief Mutable channel accessor. */                     \
  template <typename T, typename Layout>                      \
  inline T& channelName(Color<T, Layout>& c)                  \
  { return c.template channel<channelTag>(); }                \
  /*! \brief Non-mutable channel accessor. */                 \
  template <typename T, typename Layout>                      \
  inline const T& channelName(const Color<T, Layout>& c)      \
  { return c.template channel<channelTag>(); }

  DEFINE_COLOR_CHANNEL_GETTERS(red, R)
  DEFINE_COLOR_CHANNEL_GETTERS(green, G)
  DEFINE_COLOR_CHANNEL_GETTERS(blue, B)
  DEFINE_COLOR_CHANNEL_GETTERS(alpha, A)

  DEFINE_COLOR_CHANNEL_GETTERS(cyan, C)
  DEFINE_COLOR_CHANNEL_GETTERS(magenta, M)
  DEFINE_COLOR_CHANNEL_GETTERS(yellow, Y)
  DEFINE_COLOR_CHANNEL_GETTERS(black, K)

  DEFINE_COLOR_CHANNEL_GETTERS(y, Y)
  DEFINE_COLOR_CHANNEL_GETTERS(u, U)
  DEFINE_COLOR_CHANNEL_GETTERS(v, V)
#undef DEFINE_COLOR_CHANNEL_GETTERS
  //! @} ChannelAccessors

  // ======================================================================== //
  //! \defgroup ColorTraits Color and channel traits classes
  //! @{

  /*!    
    \brief Channel traits class.
  
    This is primarily intended for color conversion and color rescaling.
  */
  template <typename T>
  struct ChannelTraits
  {
    //! Channel type.
    typedef T Channel;
    //! Returns the minimum value for a specific channel type.
    static inline const T min()
    {
      return !std::numeric_limits<T>::is_integer ? 
        static_cast<T>(0) : std::numeric_limits<T>::min();
    }
    //! Returns the maximum value for a specific channel type.
    static inline const T max()
    {
      return !std::numeric_limits<T>::is_integer ? 
        static_cast<T>(1) : std::numeric_limits<T>::max();
    }
    //! Returns the minimum value for a specific channel type (in double type).
    static inline const double doubleMin()
    { return static_cast<double>(min()); }
    //! Returns the maximum value for a specific channel type (in double type).
    static inline const double doubleMax()
    { return static_cast<double>(max()); }
    //! Returns the maximum range for a specific channel type (in double type).
    static inline const double doubleRange()
    { return doubleMax() - doubleMin();}
    //! Cast function.
    template <typename U>
    static inline U cast(const T& channel)
    { return static_cast<U>(channel); }
  };
  
  // ======================================================================== //
  /*!
    \brief Color traits class.
    
    It is essentially intended for color conversion and image rescaling in order
    to view images.
   */
  template <typename Color_>
  struct ColorTraits
  {
    //! Color type.
    typedef Color_ Color;
    //! Channel type.
    typedef typename Color_::ChannelType ChannelType;
    //! Color layout type.
    typedef typename Color_::ColorLayout ColorLayout;
    //! Number of channels
    enum { NumChannels = Color_::RowsAtCompileTime };
    //! Color type where the channel type is 'double'.
    typedef DO::Color<double, ColorLayout> Color64f;
    //! Zero color (=Black for RGB).
    static inline const Color zero()
    { Color c; c.fill(ChannelType(0)); return c; }
    //! Entry-wise minimum color (=Black for RGB).
    static inline const Color min()
    { Color c; c.fill(ChannelTraits<ChannelType>::min()); return c; }
    //! Entry-wise maximum color (=White for RGB).
    static inline const Color max()
    { Color c; c.fill(ChannelTraits<ChannelType>::max()); return c; }
    //! Entry-wise minimum color (=Black for RGB).
    static inline Matrix<double, NumChannels, 1> doubleMin()
    { return min().template cast<double>(); }
    //! Entry-wise maximum color (=White for RGB).
    static inline Matrix<double, NumChannels, 1> doubleMax()
    { return max().template cast<double>(); }
    //! Entry-wise range.
    static inline Matrix<double, NumChannels, 1> doubleRange()
    { return doubleMax() - doubleMin(); }
  };

  //! \brief Specialized color traits class for MxN-Dimensional color types.
  template <typename ChannelType_, int M, int N>
  struct ColorTraits<Matrix<ChannelType_, M, N> >
  {
    //! Color type.
    typedef Matrix<ChannelType_, M, N> Color;
    //! Channel type.
    typedef ChannelType_ ChannelType;
    //! Color type where the channel type is 'double'.
    typedef Matrix<double, M, N> Color64f;
    //! Number of channels
    enum { NumChannels = M*N };
    //! Zero color (=Black for RGB).
    static inline const Color zero()
    { Color c; c.fill(ChannelType(0)); return c; }
    //! Entry-wise minimum color (=Black for RGB).
    static inline const Color min()
    { Color c; c.fill(ChannelTraits<ChannelType>::min()); return c; }
    //! Entry-wise maximum color (=White for RGB).
    static inline const Color max()
    { Color c; c.fill(ChannelTraits<ChannelType>::max()); return c; }
    //! Entry-wise minimum color (=Black for RGB).
    static inline Color64f doubleMin()
    { return min().template cast<double>(); }
    //! Entry-wise maximum color (=White for RGB).
    static inline Color64f doubleMax()
    { return max().template cast<double>(); }
    //! Entry-wise range.
    static inline Color64f doubleRange()
    { return doubleMax() - doubleMin(); }
  };

  //! Macro that defines specialized color traits classes for grayscale types.
#define DEFINE_GRAY_COLOR_TRAITS(gray_channel_t)          \
template <> struct ColorTraits<gray_channel_t>            \
{                                                         \
  /*! Color type. */                                      \
  typedef gray_channel_t Color;                           \
  /*! Channel type. */                                    \
  typedef gray_channel_t ChannelType;                     \
  /*! Double floating precision grayscale color type. */  \
  typedef double Color64f;                                \
  /*! Grayscale color layout. */                          \
  typedef Gray ColorLayout;                               \
  /*! Number of channels. */                              \
  enum { NumChannels = 1 };                               \
  /*! Zero grayscale value (=Black). */                   \
  static inline Color zero()                              \
  { return ChannelType(0); }                              \
  /*! Minimum grayscale value (=Black). */                \
  static inline Color min()                               \
  { return ChannelTraits<ChannelType>::min(); }           \
  /*! Maximum grayscale value (=White). */                \
  static inline Color max()                               \
  { return ChannelTraits<ChannelType>::max(); }           \
  /*! Double minimum grayscale value (=Black). */         \
  static inline double doubleMin()                        \
  { return ChannelTraits<ChannelType>::doubleMin(); }     \
  /*! Double maximum grayscale value (=White). */         \
  static inline double doubleMax()                        \
  { return ChannelTraits<ChannelType>::doubleMax(); }     \
  /*! Double range grayscale value. */                    \
  static inline double doubleRange()                      \
  { return doubleMax() - doubleMin(); }                   \
};

  //! Specialized color traits class for 'uchar' grayscale color.
  DEFINE_GRAY_COLOR_TRAITS(unsigned char)
  //! Specialized color traits class for 'ushort' grayscale color.
  DEFINE_GRAY_COLOR_TRAITS(unsigned short)
  //! Specialized color traits class for 'uint' grayscale color.
  DEFINE_GRAY_COLOR_TRAITS(unsigned int)
  //! Specialized color traits class for 'char' grayscale color.
  DEFINE_GRAY_COLOR_TRAITS(char)
  //! Specialized color traits class for 'short' grayscale color.
  DEFINE_GRAY_COLOR_TRAITS(short)
  //! Specialized color traits class for 'int' grayscale color.
  DEFINE_GRAY_COLOR_TRAITS(int)
  //! Specialized color traits class for 'float' grayscale color.
  DEFINE_GRAY_COLOR_TRAITS(float)
  //! Specialized color traits class for 'double' grayscale color.
  DEFINE_GRAY_COLOR_TRAITS(double)
#undef DEFINE_GRAY_COLOR_TRAITS

  //! @} ColorTraits

  
  //! \defgroup ColorConversion Color rescaling and conversion functions
  //! @{

  // ======================================================================== //
  //! Channel normalization between 0 and 1 in double floating-point precision.
  template <typename T>
  inline double getRescaledChannel64f(T value)
  {
    return (static_cast<double>(value) - ChannelTraits<T>::doubleMin())
      / ChannelTraits<T>::doubleRange();
  }
  //! Color normalization between 0 and 1 in double floating-point precision.
  template <typename T, int N>
  inline Matrix<double, N, 1> getRescaledColor64f(const Matrix<T, N, 1>& color)
  {
    Matrix<double, N, 1> min; min.fill(ChannelTraits<T>::doubleMin());
    Matrix<double, N, 1> max; max.fill(ChannelTraits<T>::doubleMax());
    return (color.template cast<double>() - min).cwiseQuotient(max - min);
  }
  //! Channel rescaling from 'double' value in [0,1] to 'T' value.
  template <typename T>
  inline void normalizeChannel(T& dst, double src)
  {
    dst = static_cast<T>(
      ColorTraits<T>::doubleMin() + src*ColorTraits<T>::doubleRange() );
  }
  //! Channel rescaling from 'T' value to 'double' value in [0,1].
  template <typename T, int N>
  inline void normalizeColor(Matrix<T, N, 1>& dst, const Matrix<double, N, 1>& src)
  {
    typedef Color<T, Rgb> Col;
    Matrix<double, N, 1> tmp( ColorTraits<Col>::doubleMin()
      + src.cwiseProduct(ColorTraits<Col>::doubleRange()) );
    dst = tmp.template cast<T>();
  }

  // ======================================================================== //
  //! RGB to grayscale color conversion function (includes color normalization).
  template <typename T, int N>
  inline double rgb2gray64f(const Matrix<T, N, 1>& rgb)
  {
    DO_STATIC_ASSERT(
      N==3 || N==4, 
      N_MUST_BE_3_OR_4_FOR_RGB_TO_GRAY_COLOR_CONVERSION);
    Matrix<double, N, 1> rgb64f(getRescaledColor64f(rgb));
    return 0.3*rgb64f[0] + 0.59*rgb64f[1] + 0.11*rgb64f[2]; 
  }
  //! Grayscale to RGB color conversion function (includes color normalization).
  template <typename T>
  inline Matrix<double, 3, 1> gray2rgb64f(T gray)
  {
    double gray64f = getRescaledChannel64f(gray);
    Matrix<double, 3, 1> rgb64f;
    rgb64f.fill(gray64f);
    return rgb64f;
  }
  //! RGB to YUV color conversion function (includes color normalization).
  template<typename T>
  inline Vector3d rgb2yuv64f(const Matrix<T, 3, 1>& rgb)
  {
    Matrix<double, 3, 1> rgb64f(getRescaledColor64f(rgb));

    double y = .299*rgb64f[0] + .587*rgb64f[1] + .114*rgb64f[2];
    return Vector3d(y, .492*(rgb64f[2]-y), .877*(rgb64f[0]-y));
  }
  //! YUV to RGB color conversion function (includes color normalization).
  template<typename T>
  inline Vector3d yuv2rgb64f(const Matrix<T, 3, 1>& yuv)
  {
    Vector3d yuv64f(getRescaledColor64f(yuv));
    double r = yuv64f[2]/.877 + yuv64f[0];
    double b = yuv64f[1]/.492 + yuv64f[0];
    double g = (yuv64f[0] - .299*r - .114*b) / .587;
    return Matrix<double, 3, 1>(r, g, b);
  }

  // ======================================================================== //
  //! Color conversion function from RGB to grayscale.
  template <typename T, typename U>
  inline void rgb2gray(T& gray, const Matrix<U, 3, 1>& rgb)
  { normalizeChannel(gray, rgb2gray64f(rgb)); }
  //! Color conversion from grayscale to RGB.
  template <typename T, typename U>
  inline void gray2rgb(Matrix<T, 3, 1>& rgb, const U& gray)
  { normalizeColor(rgb, gray2rgb64f(gray)); }
  //! Color conversion from RGB to YUV.
  template <typename T, typename U>
  inline void rgb2yuv(Matrix<T, 3, 1>& yuv, const Matrix<U, 3, 1>& rgb)
  { normalizeColor(rgb, rgb2yuv64f(yuv)); }
  //! Color conversion from YUV to RGB.
  template <typename T, typename U>
  inline void yuv2rgb(Matrix<T, 3, 1>& rgb, const Matrix<U, 3, 1>& yuv)
  { normalizeColor(rgb, yuv2rgb64f(yuv)); }

  // ======================================================================== //
  //! \brief Color conversion function with same color layout but different 
  //! channel types.
  template <typename T, typename U, typename CLayout>
  inline void convertColor(Color<T, CLayout>& dst, const Color<U, CLayout>& src)
  { normalizeColor(dst, getRescaledColor64f(src)); }
  //! \brief Color conversion from RGB to YUV.
  template <typename T, typename U>
  inline void convertColor(Color<T, Rgb>& dst, const Color<U, Yuv>& src)
  { normalizeColor(dst, yuv2rgb64f(src)); }
  //! \brief Color conversion from YUV to RGB.
  template <typename T, typename U>
  inline void convertColor(Color<T, Yuv>& dst, const Color<U, Rgb>& src)
  { normalizeColor(dst, rgb2yuv64f(src)); }

  //! Color conversion between gray and RGB color spaces.
#define COLOR_CONVERSION_BETWEEN_GRAY_AND_RGB(GrayType)             \
  /*! \brief Color conversion from RGB to Gray. */                  \
  template <typename T>                                             \
  inline void convertColor(GrayType& dst, const Color<T, Rgb>& src) \
  { rgb2gray(dst, src); }                                           \
  /*! \brief Color conversion from Gray to RGB. */                  \
  template <typename T>                                             \
  inline void convertColor(Color<T, Rgb>& dst, const GrayType& src) \
  { gray2rgb(dst, src); }
    
  COLOR_CONVERSION_BETWEEN_GRAY_AND_RGB(unsigned char)
  COLOR_CONVERSION_BETWEEN_GRAY_AND_RGB(unsigned short)
  COLOR_CONVERSION_BETWEEN_GRAY_AND_RGB(unsigned int)
  COLOR_CONVERSION_BETWEEN_GRAY_AND_RGB(char)
  COLOR_CONVERSION_BETWEEN_GRAY_AND_RGB(short)
  COLOR_CONVERSION_BETWEEN_GRAY_AND_RGB(int)
  COLOR_CONVERSION_BETWEEN_GRAY_AND_RGB(float)
  COLOR_CONVERSION_BETWEEN_GRAY_AND_RGB(double)
#undef COLOR_CONVERSION_BETWEEN_GRAY_AND_RGB

  //! Color conversion between floating point gray colors.
  inline void convertColor(float& dst, double src)
  { dst = static_cast<float>(src); }
  //! Color conversion between floating point gray colors.
  inline void convertColor(double& dst, float src)
  { dst = static_cast<double>(src); }
  //! Just in case, in order to avoid ambiguity for 'float'.
  inline void convertColor(float& dst, float src)
  { dst = src; }
  //! Just in case, in order to avoid ambiguity for 'double'.
  inline void convertColor(double& dst, double src)
  { dst = src; }

  //! Color conversion between integral gray colors.
#define DEFINE_COLOR_CONVERSION_BETWEEN_INTEGRAL_GRAY_TYPES(Gray1, Gray2) \
  /*! \brief Color conversion between integral gray colors. */            \
  inline void convertColor(Gray2& dst, Gray1 src)                         \
  { normalizeChannel(dst, getRescaledChannel64f(src)); }                  \
  /*! \brief Color conversion between integral gray colors. */            \
  inline void convertColor(Gray1& dst, Gray2 src)                         \
  { normalizeChannel(dst, getRescaledChannel64f(src)); }

  // uchar
  DEFINE_COLOR_CONVERSION_BETWEEN_INTEGRAL_GRAY_TYPES(unsigned char, unsigned short)
  DEFINE_COLOR_CONVERSION_BETWEEN_INTEGRAL_GRAY_TYPES(unsigned char, unsigned int)
  DEFINE_COLOR_CONVERSION_BETWEEN_INTEGRAL_GRAY_TYPES(unsigned char, char)
  DEFINE_COLOR_CONVERSION_BETWEEN_INTEGRAL_GRAY_TYPES(unsigned char, short)
  DEFINE_COLOR_CONVERSION_BETWEEN_INTEGRAL_GRAY_TYPES(unsigned char, int)
  // ushort
  DEFINE_COLOR_CONVERSION_BETWEEN_INTEGRAL_GRAY_TYPES(unsigned short, unsigned int)
  DEFINE_COLOR_CONVERSION_BETWEEN_INTEGRAL_GRAY_TYPES(unsigned short, int)
  DEFINE_COLOR_CONVERSION_BETWEEN_INTEGRAL_GRAY_TYPES(unsigned short, char)
  DEFINE_COLOR_CONVERSION_BETWEEN_INTEGRAL_GRAY_TYPES(unsigned short, short)
  // uint
  DEFINE_COLOR_CONVERSION_BETWEEN_INTEGRAL_GRAY_TYPES(unsigned int, int)
  DEFINE_COLOR_CONVERSION_BETWEEN_INTEGRAL_GRAY_TYPES(unsigned int, char)
  DEFINE_COLOR_CONVERSION_BETWEEN_INTEGRAL_GRAY_TYPES(unsigned int, short)
  // char
  DEFINE_COLOR_CONVERSION_BETWEEN_INTEGRAL_GRAY_TYPES(char, short)
  DEFINE_COLOR_CONVERSION_BETWEEN_INTEGRAL_GRAY_TYPES(char, int)
  // short
  DEFINE_COLOR_CONVERSION_BETWEEN_INTEGRAL_GRAY_TYPES(short, int)
  // int
#undef DEFINE_COLOR_CONVERSION_BETWEEN_INTEGRAL_GRAY_TYPES

  //! Color conversion between integral and floating point gray colors.
#define DEFINE_COLOR_CONVERSION_BETWEEN_INT_AND_FLOATING_TYPES(Float, Int)\
  /*! \brief Color conversion between integral and floating point gray. */\
  inline void convertColor(Float& dst, Int src)                           \
  {                                                                       \
    Float M = static_cast<Float>(ColorTraits<Int>::max());                \
    Float m = static_cast<Float>(ColorTraits<Int>::min());                \
    dst = (static_cast<Float>(src)-m) / (M-m);                            \
  }                                                                       \
  /*! \brief Color conversion between integral and floating point gray. */\
  inline void convertColor(Int& dst, Float src)                           \
  {                                                                       \
    src = Float(ColorTraits<Int>::min())                                  \
      + src*( Float(ColorTraits<Int>::max())                              \
          - Float(ColorTraits<Int>::min()) );                             \
    dst = static_cast<Int>(src);                                          \
  }

  DEFINE_COLOR_CONVERSION_BETWEEN_INT_AND_FLOATING_TYPES(float, unsigned char)
  DEFINE_COLOR_CONVERSION_BETWEEN_INT_AND_FLOATING_TYPES(float, unsigned short)
  DEFINE_COLOR_CONVERSION_BETWEEN_INT_AND_FLOATING_TYPES(float, unsigned int)
  DEFINE_COLOR_CONVERSION_BETWEEN_INT_AND_FLOATING_TYPES(float, char)
  DEFINE_COLOR_CONVERSION_BETWEEN_INT_AND_FLOATING_TYPES(float, short)
  DEFINE_COLOR_CONVERSION_BETWEEN_INT_AND_FLOATING_TYPES(float, int)

  DEFINE_COLOR_CONVERSION_BETWEEN_INT_AND_FLOATING_TYPES(double, unsigned char)
  DEFINE_COLOR_CONVERSION_BETWEEN_INT_AND_FLOATING_TYPES(double, unsigned short)
  DEFINE_COLOR_CONVERSION_BETWEEN_INT_AND_FLOATING_TYPES(double, unsigned int)
  DEFINE_COLOR_CONVERSION_BETWEEN_INT_AND_FLOATING_TYPES(double, char)
  DEFINE_COLOR_CONVERSION_BETWEEN_INT_AND_FLOATING_TYPES(double, short)
  DEFINE_COLOR_CONVERSION_BETWEEN_INT_AND_FLOATING_TYPES(double, int)
#undef DEFINE_COLOR_CONVERSION_BETWEEN_INT_AND_FLOATING_TYPES

  //! Color conversion from RGBA to RGB
  template <typename T>
  void convertColor(Color<T, Rgb>& dst, const Color<T, Rgba>& src)
  {
    red(dst) = red(src);
    blue(dst) = blue(src);
    green(dst) = green(src);
  }
  //! Color conversion from RGB to RGBA
  template <typename T>
  void convertColor(Color<T, Rgba>& dst, const Color<T, Rgb>& src)
  {
    red(dst) = red(src);
    blue(dst) = blue(src);
    green(dst) = green(src);
    alpha(dst) = ChannelTraits<T>::max();
  }
  //! Color conversion from RGB to RGBA
  template <typename T, typename U>
  void convertColor(Color<T, Rgb>& dst, const Color<U, Rgba>& src)
  {
    convertColor(red(dst), red(src));
    convertColor(blue(dst), blue(src));
    convertColor(green(dst), green(src));
  }
  //! Color conversion from RGBA to RGB
  template <typename T, typename U>
  void convertColor(Color<T, Rgba>& dst, const Color<U, Rgb>& src)
  {
    convertColor(red(dst), red(src));
    convertColor(blue(dst), blue(src));
    convertColor(green(dst), green(src));
    alpha(dst) = ChannelTraits<T>::max();
  }
  //! Color conversion from RGBA to unsigned char
  template <typename T, typename U>
  void convertColor(T& gray, const Color<U, Rgba>& src)
  {
    Color<double, Rgb> rgb64f;
    convertColor(rgb64f, src);
    convertColor(gray, rgb64f);
  }

  //! @} ColorConversion



  //! \defgroup ColorTypes Color typedefs
  //! @{

  // ======================================================================== //
  //! self-explanatory.
  typedef unsigned char gray8;
  //! self-explanatory.
  typedef char gray8s;
  //! self-explanatory.
  typedef unsigned short gray16;
  //! self-explanatory.
  typedef short gray16s;
  //! self-explanatory.
  typedef unsigned int gray32;
  //! self-explanatory.
  typedef int gray32s;
  //! self-explanatory.
  typedef float gray32f;
  //! self-explanatory.
  typedef double gray64f;

  // ======================================================================== //
  //! Macro for generic color typedefs
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
  typedef Color<unsigned char, colorspace> colorspace##8;   \
  /*! \brief {ColorSpace}{BitDepthPerChannel} */            \
  typedef Color<unsigned short, colorspace> colorspace##16; \
  /*! \brief {ColorSpace}{BitDepthPerChannel} */            \
  typedef Color<unsigned int, colorspace> colorspace##32;   \
  /*! \brief {ColorSpace}{BitDepthPerChannel} */            \
  typedef Color<char, colorspace> colorspace##8s;           \
  /*! \brief {ColorSpace}{BitDepthPerChannel} */            \
  typedef Color<short, colorspace> colorspace##16s;         \
  /*! \brief {ColorSpace}{BitDepthPerChannel} */            \
  typedef Color<int, colorspace> colorspace##32s;           \
  /*! \brief {ColorSpace}{BitDepthPerChannel} */            \
  typedef Color<float, colorspace> colorspace##32f;         \
  /*! \brief {ColorSpace}{BitDepthPerChannel} */            \
  typedef Color<double, colorspace> colorspace##64f;

  DEFINE_COLOR_TYPES(Rgb)
  DEFINE_COLOR_TYPES(Rgba)
  DEFINE_COLOR_TYPES(Cmyk)
  DEFINE_COLOR_TYPES(Yuv)
#undef DEFINE_COLOR_TYPES
  //! @} ColorTypes

  // ======================================================================== //
  //! \defgroup PrimaryColors Primary Colors 
  //! @{

  //! White color function.
  template <typename T> inline Matrix<T, 3, 1> white()
  { 
    return Matrix<T,3,1>(
      ChannelTraits<T>::max(),
      ChannelTraits<T>::max(),
      ChannelTraits<T>::max() ); 
  }
  //! Black color function.
  template <typename T> inline Matrix<T, 3, 1> black()
  { 
    return Matrix<T,3,1>(
      ChannelTraits<T>::min(),
      ChannelTraits<T>::min(),
      ChannelTraits<T>::min() ); 
  }
  //! Red color function.
  template <typename T> inline Matrix<T, 3, 1> red()
  {
    return Matrix<T,3,1>(
      ChannelTraits<T>::max(),
      ChannelTraits<T>::min(),
      ChannelTraits<T>::min() ); 
  }
  //! Green color function.
  template <typename T> inline Matrix<T, 3, 1> green()
  {
    return Matrix<T,3,1>(
      ChannelTraits<T>::min(),
      ChannelTraits<T>::max(),
      ChannelTraits<T>::min() );
  }
  //! Blue color function.
  template <typename T> inline Matrix<T, 3, 1> blue()
  {
    return Matrix<T,3,1>(
      ChannelTraits<T>::min(),
      ChannelTraits<T>::min(),
      ChannelTraits<T>::max() ); 
  }
  //! Cyan color function.
  template <typename T> inline Matrix<T, 3, 1> cyan()
  {
    return Matrix<T,3,1>(
      ChannelTraits<T>::min(),
      ChannelTraits<T>::max(),
      ChannelTraits<T>::max() ); 
  }
  //! Yellow color function.
  template <typename T> inline Matrix<T, 3, 1> yellow()
  {
    return Matrix<T,3,1>(
      ChannelTraits<T>::max(),
      ChannelTraits<T>::max(),
      ChannelTraits<T>::min() ); 
  }
  //! Magenta color function.
  template <typename T> inline Matrix<T, 3, 1> magenta()
  {
    return Matrix<T,3,1>(
      ChannelTraits<T>::max(),
      ChannelTraits<T>::min(),
      ChannelTraits<T>::max() ); 
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

  //! @}

} /* namespace DO */

#endif /* DO_CORE_COLOR_HPP */