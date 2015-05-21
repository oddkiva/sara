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

#ifndef DO_SARA_CORE_PIXEL_PIXEL_HPP
#define DO_SARA_CORE_PIXEL_PIXEL_HPP


#include <functional>
#include <stdexcept>

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Meta.hpp>


// Pixel data structures.
namespace DO {

  //! \brief Simple pixel class implemented as a vector.
  template <typename _T, typename _ColorSpace>
  class Pixel : public Matrix<_T, _ColorSpace::size, 1>
  {
    typedef Matrix<_T, _ColorSpace::size, 1> base_type;

  public:
    //! Channel type.
    typedef _T channel_type;
    typedef _ColorSpace color_space_type;

    //! Default constructor.
    inline Pixel()
      : base_type()
    {
    }

    //! Custom constructor.
    inline Pixel(_T x, _T y, _T z)
      : base_type(x, y, z)
    {
    }

    //! Custom constructor.
    inline Pixel(_T x, _T y, _T z, _T t)
      : base_type(x, y, z, t)
    {
    }

    //! Copy constructor.
    template<typename OtherDerived>
    inline Pixel(const Eigen::MatrixBase<OtherDerived>& other)
      : base_type(other)
    {
    }

    //! Assignment operator.
    template <typename _OtherDerived>
    inline Pixel& operator=(const Eigen::MatrixBase<_OtherDerived>& other)
    {
      this->base_type::operator=(other); return *this;
    }

    //! Constant channel access function.
    template <typename _ChannelTag>
    inline const channel_type& channel() const
    {
      return (*this)[Meta::IndexOf<_ColorSpace, _ChannelTag>::value];
    }

    //! Mutable channel access function.
    template <typename _ChannelTag>
    inline channel_type& channel()
    {
      return (*this)[Meta::IndexOf<_ColorSpace, _ChannelTag>::value];
    }

    //! Returns the number of channels.
    int num_channels() const
    {
      return _ColorSpace::size;
    }
  };

}


#endif /* DO_SARA_CORE_PIXEL_PIXEL_HPP */