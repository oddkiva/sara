// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <functional>
#include <stdexcept>

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Meta.hpp>
#include <DO/Sara/Core/Pixel/ColorSpace.hpp>


// Pixel data structures.
namespace DO { namespace Sara {

  //! @brief 3D packed pixel.
  template <typename _BitField, int _Sz0, int _Sz1, int _Sz2>
  struct PackedPixelBase_3
  {
    using bitfield_type = _BitField;
    enum { num_channels = 3 };
    _BitField channel_0 : _Sz0;
    _BitField channel_1 : _Sz1;
    _BitField channel_2 : _Sz2;

    template <typename Op>
    inline void apply_op(const PackedPixelBase_3& other)
    {
      Op op;
      channel_0 = op(channel_0, other.channel_0);
      channel_1 = op(channel_1, other.channel_1);
      channel_2 = op(channel_2, other.channel_2);
    }

    inline bool operator==(const PackedPixelBase_3& other) const
    {
      return
        channel_0 == other.channel_0 &&
        channel_1 == other.channel_1 &&
        channel_2 == other.channel_2;
    }

    inline bool operator!=(const PackedPixelBase_3& other) const
    {
      return !(*this == other);
    }
  };


  //! @brief 4D packed pixel.
  template <typename _BitField, int _Sz0, int _Sz1, int _Sz2, int _Sz3>
  struct PackedPixelBase_4
  {
    using bitfield_type = _BitField;
    enum { num_channels = 4 };
    _BitField channel_0 : _Sz0;
    _BitField channel_1 : _Sz1;
    _BitField channel_2 : _Sz2;
    _BitField channel_3 : _Sz3;

    template <typename Op>
    inline void apply_op(const PackedPixelBase_4& other)
    {
      Op op;
      channel_0 = op(channel_0, other.channel_0);
      channel_1 = op(channel_1, other.channel_1);
      channel_2 = op(channel_2, other.channel_2);
      channel_3 = op(channel_3, other.channel_3);
    }

    inline bool operator==(const PackedPixelBase_4& other) const
    {
      return
        channel_0 == other._channel_0 &&
        channel_1 == other._channel_1 &&
        channel_2 == other._channel_2 &&
        channel_3 == other._channel_3;
    }

    inline bool operator!=(const PackedPixelBase_4& other) const
    { return !(*this == other);}
  };


  //! @brief Channel getter.
  template <typename _PackedPixel, int _Index> struct Channel;

#define SPECIALIZE_CHANNEL(index)                               \
  template <typename _PackedPixel>                              \
  struct Channel<_PackedPixel, index>                           \
  {                                                             \
    Channel(_PackedPixel& p) : _p(p) {}                         \
    using bitfield_type = typename _PackedPixel::bitfield_type; \
    bitfield_type operator=(bitfield_type value)                \
    { _p.channel_##index = value; }                             \
    bool operator==(bitfield_type value) const                  \
    { return _p.channel_##index == value; }                     \
    _PackedPixel& _p;                                           \
  };

  SPECIALIZE_CHANNEL(0)
  SPECIALIZE_CHANNEL(1)
  SPECIALIZE_CHANNEL(2)
  SPECIALIZE_CHANNEL(3)


  //! @brief Color model.
  template <typename _ColorSpace, typename _ChannelOrder>
  struct ColorModel
  {
    using color_space_type = _ColorSpace;
    using channel_order_type = _ChannelOrder;
  };

  //! @brief Channel index getter.
  template <typename _ColorModel, typename _ChannelTag>
  struct ChannelIndex
  {
    using color_space_type = typename _ColorModel::color_space_type;
    using channel_order_type = typename _ColorModel::channel_order_type;
    using channel_tag_type = _ChannelTag;

    enum
    {
      channel = Meta::IndexOf<color_space_type, _ChannelTag>::value,
      value = Meta::Get<channel_order_type, channel>::value
    };
  };

  //! @brief PackedPixel class.
  template <typename _PackedPixelBase, typename _ColorModel>
  class PackedPixel: protected _PackedPixelBase
  {
    using base_type = _PackedPixelBase;
    using self_type = PackedPixel;

  public:
    using bitfield_type = typename base_type::bitfield_type;
    using color_layout_type = _ColorModel;
    using color_space_type = typename _ColorModel::color_space_type;
    using channel_order_type = typename _ColorModel::channel_order_type;

  public:
    PackedPixel() = default;

    inline PackedPixel(bitfield_type v0, bitfield_type v1, bitfield_type v2)
    {
      base_type::channel_0 = v0;
      base_type::channel_1 = v1;
      base_type::channel_2 = v2;
    }

    inline PackedPixel(bitfield_type v0, bitfield_type v1, bitfield_type v2,
                       bitfield_type v3)
    {
      base_type::channel_0 = v0;
      base_type::channel_1 = v1;
      base_type::channel_2 = v2;
      base_type::channel_3 = v3;
    }

    inline PackedPixel(const PackedPixel& other)
    {
      *this = other;
    }

    inline self_type operator+(const self_type& other) const
    {
      self_type result(*this);
      result += other;
      return result;
    }

    inline self_type operator-(const self_type& other) const
    {
      self_type result(*this);
      result -= other;
      return result;
    }

    inline self_type operator*(bitfield_type scalar) const
    {
      self_type result(*this);
      result *= scalar;
      return result;
    }

    friend inline self_type operator*(bitfield_type u, const self_type& v)
    {
      return v*u;
    }

    inline void operator+=(const self_type& other)
    {
      this->template apply_op<std::plus<bitfield_type> >(other);
    }

    inline void operator-=(const self_type& other)
    {
      this->template apply_op<std::minus<bitfield_type> >(other);
    }

    inline void operator*=(bitfield_type other)
    {
      this->template apply_op<std::multiplies<bitfield_type> >(
        self_type(other, other, other)
      );
    }

    bool operator==(const self_type& other) const
    {
      return base_type::operator==(other);
    }

    bool operator!=(const self_type& other) const
    {
      return base_type::operator!=(other);
    }

    template <typename _ChannelTag>
    Channel<_PackedPixelBase, ChannelIndex<_ColorModel, _ChannelTag>::value>
    channel()
    {
      return Channel<
        _PackedPixelBase,
        ChannelIndex<_ColorModel, _ChannelTag>::value
      >(*this);
    }
  };


} /* namespace Sara */
} /* namespace DO */
