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

#ifndef DO_CORE_IMAGE_IMAGE_HPP
#define DO_CORE_IMAGE_IMAGE_HPP


#include <DO/Core/Image/ElementTraits.hpp>
#include <DO/Core/MultiArray.hpp>


namespace DO {

  // ======================================================================== //
  /*!
    \ingroup Core
    \defgroup Image Image
    @{
   */

  //! \brief Forward declaration of the image class.
  template <typename Color, int N = 2> class Image;

  //! \brief Forward declaration of the generic image converter class.
  template <typename T, typename U, int N>
  void convert_channel(const Image<T, N>& src, Image<U, N>& dst);

  //! \brief Convert color of image.
  template <typename T, typename U, int N>
  void convert_color(const Image<T, N>& src, Image<U, N>& dst);

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
      : base_type()
    {
    }

    //! Constructor with specified sizes.
    inline explicit Image(const vector_type& sizes)
      : base_type(sizes)
    {
    }

    //! Constructor which wraps raw data.
    inline Image(Color *data, const vector_type& sizes,
                 bool acquire_ownership = false)
      : base_type(data, sizes, acquire_ownership)
    {
    }

    //! Constructor with specified sizes.
    inline Image(int width, int height)
      : base_type(width, height)
    {
    }

    //! Constructor with specified sizes.
    inline Image(int width, int height, int depth)
      : base_type(width, height, depth)
    {
    }

    //! Copy constructor.
    inline Image(const base_type& other)
      : base_type(other)
    {
    }

    //! Constant width getter.
    inline int width() const
    {
      return this->base_type::rows();
    }

    //! Constant height getter.
    inline int height() const
    {
      return this->base_type::cols();
    }

    //! Constant depth getter, which is only valid for 3D images.
    inline int depth() const
    {
      return this->base_type::depth();
    }

    //! Color conversion method.
    template <typename Color2>
    Image<Color2, N> convert_color() const
    {
      Image<Color2, N> dst(base_type::sizes());
      DO::convert_color(*this, dst);
      return dst;
    }

    //! Color conversion method.
    template <typename Color2>
    Image<Color2, N> convert_channel() const
    {
      Image<Color2, N> dst(base_type::sizes());
      DO::convert_channel(*this, dst);
      return dst;
    }

    //! Convenient helper for chaining filters.
    template <template<typename, int> class Filter>
    inline typename Filter<Color, N>::ReturnType compute() const
    {
      return Filter<Color, N>(*this)();
    }

    template <template<typename, int> class Filter>
    inline typename Filter<Color, N>::return_type
    compute(const typename Filter<Color, N>::parameter_type& param) const
    {
      return Filter<Color, N>(*this)(param);
    }
  };


  //! @}

} /* namespace DO */


#endif /* DO_CORE_IMAGE_IMAGE_HPP */