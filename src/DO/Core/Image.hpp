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

#ifndef DO_CORE_IMAGE_HPP
#define DO_CORE_IMAGE_HPP

#include "Color.hpp"
#include "MultiArray.hpp"

namespace DO {

  // ======================================================================== //
  /*!
    \ingroup Core
    \defgroup Image Image
    @{
   */

  //! \brief The specialized element traits class when the entry is a color.
  template <typename T, typename Layout>
  struct ElementTraits<Pixel<T, Layout> >
  {
    typedef Array<T, Layout::size, 1> value_type; //!< STL-like typedef.
    typedef size_t size_type; //!< STL-like typedef.
    typedef value_type * pointer; //!< STL-like typedef.
    typedef const value_type * const_pointer; //!< STL-like typedef.
    typedef value_type& reference; //!< STL-like typedef.
    typedef const value_type& const_reference; //!< STL-like typedef.
    typedef value_type * iterator; //!< STL-like typedef.
    typedef const value_type * const_iterator; //!< STL-like typedef.
    static const bool is_scalar = false; //!< STL-like typedef.
  };

  //! The forward declaration of the image class.
  template <typename Color, int N = 2> class Image;

  //! \brief Helper function for color conversion.
  template <typename T, typename U, int N>
  void convert(Image<T, N>& dst, const Image<U, N>& src);

  //! \brief The image class.
  template <typename Color, int N>
  class Image : public MultiArray<Color, N, ColMajor>
  {
    typedef MultiArray<Color, N, ColMajor> base_type;

  public: /* interface */
    //! N-dimensional integral vector type.
    typedef typename base_type::vector_type vector_type, Vector;
    
    //! Default constructor.
    inline Image()
      : base_type() {}

    //! Constructor with specified sizes.
    inline explicit Image(const vector_type& sizes)
      : base_type(sizes) {}

    //! Constructor which wraps raw data.
    inline Image(Color *data, const vector_type& sizes,
                 bool getOwnership = false)
      : base_type(data, sizes, getOwnership) {}

    //! Constructor with specified sizes.
    inline Image(int width, int height)
      : base_type(width, height) {}

    //! Constructor with specified sizes.
    inline Image(int width, int height, int depth)
      : base_type(width, height, depth) {}

    //! Copy constructor.
    inline Image(const base_type& x)
      : base_type(x) {}

    //! Assignment operators.
    inline const Image& operator=(const Image& I)
    { base_type::operator=(I); return *this;}

    //! Constant width getter.
    inline int width() const { return this->base_type::rows(); }

    //! Constant height getter.
    inline int height() const {  return this->base_type::cols(); }

    //! Constant depth getter, which is only valid for 3D images.
    inline int depth() const {  return this->base_type::depth(); }

    //! Color conversion method.
    template <typename Color2>
    Image<Color2, N> convert() const
    {
      Image<Color2, N> dst(base_type::sizes());
      DO::convert(dst, *this);
      return dst;
    }

    //! Convenient helper for chaining filters.
    template <template<typename, int> class Filter>
    inline typename Filter<Color, N>::ReturnType
    compute() const
    { return Filter<Color, N>(*this)(); }

    template <template<typename, int> class Filter>
    inline typename Filter<Color, N>::ReturnType
    compute(const typename Filter<Color, N>::ParamType& param) const
    { return Filter<Color, N>(*this)(param); }
  };


  // ====================================================================== //
  // Generic image conversion function.
  //! \brief Generic image converter class.
  template <typename T, typename U, int N>
  struct ConvertImage {
    //! Implementation of the image conversion.
    static void apply(Image<T, N>& dst, const Image<U, N>& src)
    {
      if (dst.sizes() != src.sizes())
        dst.resize(src.sizes());

      const U *src_first = src.data();
      const U *src_last = src_first + src.size();

      T *dst_first  = dst.data();

      for ( ; src_first != src_last; ++src_first, ++dst_first)
        convert_color(*dst_first, *src_first);
    }
  };

  //! \brief Specialized image converter class when the source and color types
  //! are the same.
  template <typename T, int N>
  struct ConvertImage<T,T,N> {
    //! Implementation of the image conversion.
    static void apply(Image<T, N>& dst, const Image<T, N>& src)
    {
      dst = src;
    }
  };

  template <typename T, typename U, int N>
  inline void convert(Image<T, N>& dst, const Image<U, N>& src)
  {
    ConvertImage<T,U,N>::apply(dst, src);
  }


  // ====================================================================== //
  // Find min and max values in images according to point-wise comparison.
  //! \brief Find min and max pixel values of the image.
  template <typename T, int N, typename Layout>
  void findMinMax(Pixel<T, Layout>& min, Pixel<T, Layout>& max,
                  const Image<Pixel<T, Layout>, N>& src)
  {
    const Pixel<T,Layout> *src_first = src.data();
    const Pixel<T,Layout> *src_last = src_first + src.size();

    min = *src_first;
    max = *src_first;

    for ( ; src_first != src_last; ++src_first)
    {
      min = min.cwiseMin(*src_first);
      max = max.cwiseMax(*src_first);
    }
  }

  //! Macro that defines min-max value functions for a specific grayscale 
  //! color types.
#define DEFINE_FINDMINMAX_GRAY(T)                               \
  /*! \brief Find min and max grayscale values of the image. */ \
  template <int N>                                              \
  inline void findMinMax(T& min, T& max, const Image<T, N>& src)\
  {                                                             \
    const T *src_first = src.data();                            \
    const T *src_last = src_first + src.size();                 \
                                                                \
    min = *std::min_element(src_first, src_last);               \
    max = *std::max_element(src_first, src_last);               \
  }

  DEFINE_FINDMINMAX_GRAY(unsigned char)
  DEFINE_FINDMINMAX_GRAY(char)
  DEFINE_FINDMINMAX_GRAY(unsigned short)
  DEFINE_FINDMINMAX_GRAY(short)
  DEFINE_FINDMINMAX_GRAY(unsigned int)
  DEFINE_FINDMINMAX_GRAY(int)
  DEFINE_FINDMINMAX_GRAY(float)
  DEFINE_FINDMINMAX_GRAY(double)
#undef DEFINE_FINDMINMAX_GRAY


  // ====================================================================== //
  // Image rescaling functions
  //! \brief color rescaling function.
  template <typename T, typename Layout, int N>
  inline Image<Pixel<T,Layout>, N> colorRescale(
    const Image<Pixel<T,Layout>, N>& src,
    const Pixel<T, Layout>& a = black<T>(),
    const Pixel<T, Layout>& b = white<T>())
  {
    Image<Pixel<T,Layout>, N> dst(src.sizes());

    const Pixel<T,Layout> *src_first = src.data();
    const Pixel<T,Layout> *src_last = src_first + src.size();
    Pixel<T,Layout> *dst_first  = dst.data();

    Pixel<T,Layout> min(*src_first);
    Pixel<T,Layout> max(*src_first);
    for ( ; src_first != src_last; ++src_first)
    {
      min = min.cwiseMin(*src_first);
      max = max.cwiseMax(*src_first);
    }

    if (min == max)
    {
      std::cerr << "Warning: min == max!" << std::endl;
      return dst;
    }

    for (src_first = src.data(); src_first != src_last; 
       ++src_first, ++dst_first)
      *dst_first = a + (*src_first-min).cwiseProduct(b-a).
                                        cwiseQuotient(max-min);

    return dst;
  }

  //! \brief Return min color value.
  template <typename T>
  inline T color_min_value()
  {
    return channel_min_value<T>();
  }

  //! \brief Return max color value.
  template <typename T>
  inline T color_max_value()
  {
    return channel_max_value<T>();
  }

  //! \brief Return min color value.
  template <typename T, int N>
  inline Matrix<T, N, 1> color_min_value()
  {
    Matrix<T, N, 1> min;
    min.fill(channel_min_value<T>());
    return min;
  }
  
  //! \brief Return max color value.
  template <typename T, int N>
  inline Matrix<T, N, 1> color_max_value()
  {
    Matrix<T, N, 1> min;
    min.fill(channel_min_value<T>());
    return min;
  }

  //! \brief Rescales color values properly for viewing purposes.
  template <typename T, int N>
  inline Image<T, N> colorRescale(const Image<T, N>& src,
                                  const T& a = color_min_value<T>(),
                                  const T& b = color_max_value<T>())
  {
    Image<T, N> dst(src.sizes());

    const T *src_first = src.data();
    const T *src_last = src_first + src.size();
    T *dst_first  = dst.data();

    T min = *std::min_element(src_first, src_last);
    T max = *std::max_element(src_first, src_last);

    if (min == max)
      throw std::runtime_error("Error: cannot rescale image! min == max");

    for ( ; src_first != src_last; ++src_first, ++dst_first)
      *dst_first = a + (b-a)*(*src_first-min)/(max-min);

    return dst;
  }

  //! \brief color rescaling functor helper.
  template <typename T, int N>
  struct ColorRescale
  {
    typedef Image<T, N> ReturnType;
    ColorRescale(const Image<T, N>& src) : src_(src) {}
    ReturnType operator()() const { return colorRescale(src_); }
    const Image<T, N>& src_;
  };

  //! @}

} /* namespace DO */

#endif /* DO_CORE_IMAGE_HPP */