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

#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Meta.hpp>

#include <functional>
#include <stdexcept>


// Pixel data structures.
namespace DO { namespace Sara {

  //! @brief Simple pixel class implemented as a vector.
  template <typename _T, typename _ColorSpace>
  class Pixel : public Matrix<_T, _ColorSpace::size, 1>
  {
    using base_type = Matrix<_T, _ColorSpace::size, 1>;

  public:
    //! Channel type.
    using channel_type = _T;
    using color_space_type = _ColorSpace;

    //! Default constructor.
    inline Pixel()
      : base_type{}
    {
    }

    //! Custom constructor.
    inline Pixel(_T x, _T y, _T z)
      : base_type{ x, y, z }
    {
    }

    //! Custom constructor.
    inline Pixel(_T x, _T y, _T z, _T t)
      : base_type{ x, y, z, t }
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
    inline auto operator=(const Eigen::MatrixBase<_OtherDerived>& other) -> Pixel&
    {
      this->base_type::operator=(other);
      return *this;
    }

    //! Constant channel access function.
    template <typename _ChannelTag>
    inline auto channel() const -> const channel_type&
    {
      return (*this)[Meta::IndexOf<_ColorSpace, _ChannelTag>::value];
    }

    //! Mutable channel access function.
    template <typename _ChannelTag>
    inline auto channel() -> channel_type&
    {
      return (*this)[Meta::IndexOf<_ColorSpace, _ChannelTag>::value];
    }

    //! Returns the number of channels.
    static constexpr auto num_channels() -> int
    {
      return _ColorSpace::size;
    }
  };

} /* namespace Sara */
} /* namespace DO */
