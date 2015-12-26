// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_SARA_CORE_PIXEL_PACKEDPIXEL_HPP
#define DO_SARA_CORE_PIXEL_PACKEDPIXEL_HPP


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
    typedef _BitField bitfield_type;
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
    { return !(*this == other);}
  };


  //! @brief 4D packed pixel.
  template <typename _BitField, int _Sz0, int _Sz1, int _Sz2, int _Sz3>
  struct PackedPixelBase_4
  {
    typedef _BitField bitfield_type;
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
    typedef typename _PackedPixel::bitfield_type bitfield_type; \
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
    typedef _ColorSpace color_space_type;
    typedef _ChannelOrder channel_order_type;
  };

  //! @brief Channel index getter.
  template <typename _ColorModel, typename _ChannelTag>
  struct ChannelIndex
  {
    typedef typename _ColorModel::color_space_type color_space_type;
    typedef typename _ColorModel::channel_order_type channel_order_type;
    typedef _ChannelTag channel_tag_type;

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
    typedef _PackedPixelBase base_type;
    typedef PackedPixel self_type;

  public:
    typedef typename base_type::bitfield_type bitfield_type;
    typedef _ColorModel color_layout_type;
    typedef typename _ColorModel::color_space_type color_space_type;
    typedef typename _ColorModel::channel_order_type channel_order_type;

  public:
    inline PackedPixel()
    {
    }

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


#endif /* DO_SARA_CORE_PIXEL_PACKEDPIXEL_HPP */
