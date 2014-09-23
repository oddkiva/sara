#pragma once

#include <functional>
#include <stdexcept>

#include <DO/Core/EigenExtension.hpp>
#include <DO/Core/Meta.hpp>


// Pixel data structures.
namespace DO {

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
    inline Pixel(const base_type& x) : base_type(x) {}

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