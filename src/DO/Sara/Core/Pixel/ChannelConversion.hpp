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

#ifndef DO_SARA_CORE_PIXEL_CHANNELCONVERSION_HPP
#define DO_SARA_CORE_PIXEL_CHANNELCONVERSION_HPP


#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Pixel/Pixel.hpp>


// Channel conversion from a type to another.
namespace DO { namespace Sara {

  //! \brief Return minimum value for channel of type 'T'.
  template <typename T>
  inline T channel_min_value()
  {
    using std::numeric_limits;
    return numeric_limits<T>::is_integer ? numeric_limits<T>::min() : T(0);
  }

  //! \brief Return maximum value for channel of type 'T'.
  template <typename T>
  inline T channel_max_value()
  {
    using std::numeric_limits;
    return numeric_limits<T>::is_integer ? numeric_limits<T>::max() : T(1);
  }

  //! \brief Convert integral channel value to floating-point value.
  template <typename Int, typename Float>
  inline Float to_normalized_float_channel(Int src)
  {
    static_assert(
      std::numeric_limits<Int>::is_integer,
      "Channel conversion must be from integer type to floating point type");

    using std::numeric_limits;
    const auto float_min = static_cast<Float>(numeric_limits<Int>::min());
    const auto float_max = static_cast<Float>(numeric_limits<Int>::max());
    const auto float_range = float_max - float_min;
    return (static_cast<Float>(src) - float_min) / float_range;
  }

  //! \brief Convert floating-point channel value to integer value.
  template <typename Int, typename Float>
  inline Int to_rescaled_integral_channel(Float src)
  {
    static_assert(
      std::numeric_limits<Int>::is_integer,
      "Channel conversion must be from floating point type to integer type");

    using std::numeric_limits;
    auto float_min = static_cast<Float>(numeric_limits<Int>::min());
    auto float_max = static_cast<Float>(numeric_limits<Int>::max());
    auto float_range = float_max - float_min;
    src = float_min + src * float_range;

    const auto delta_max = std::abs(src - float_max) / float_range;
    const auto delta_min = std::abs(src - float_min) / float_range;
    const auto eps = sizeof(Float) == 4 ?
      Float(1e-5) : // i.e., if 'Float' == 'float'.
      Float(1e-9);  // i.e., if 'Float' == 'double'.

    if (delta_max <= eps)
      return std::numeric_limits<Int>::max();

    if (delta_min <= eps)
      return std::numeric_limits<Int>::min();

    return static_cast<Int>(floor(src + 0.5));
  }

} /* namespace Sara */
} /* namespace DO */


// Unified API for channel conversion.
namespace DO { namespace Sara {

  //! \brief Convert a double gray value to a float gray value.
  inline void convert_channel(double src, float& dst)
  {
    dst = static_cast<float>(src);
  }

  //! \brief Convert a float gray value to a double gray value.
  inline void convert_channel(float src, double& dst)
  {
    dst = static_cast<double>(src);
  }

  //! \brief Convert an integer gray value to a float gray value.
  template <typename Int>
  inline void convert_channel(Int src, float& dst)
  {
    dst = to_normalized_float_channel<Int, float>(src);
  }

  //! \brief Convert an integer gray value to a double gray value.
  template <typename Int>
  inline void convert_channel(Int src, double& dst)
  {
    dst = to_normalized_float_channel<Int, double>(src);
  }

  //! \brief Convert a float gray value to an integer gray value.
  template <typename Int>
  inline void convert_channel(float src, Int& dst)
  {
    dst = to_rescaled_integral_channel<Int, float>(src);
  }

  //! \brief Convert a double gray value to a integer gray value.
  template <typename Int>
  inline void convert_channel(double src, Int& dst)
  {
    dst = to_rescaled_integral_channel<Int, double>(src);
  }

  //! \brief Convert an integer gray value to another one.
  template <typename SrcInt, typename DstInt>
  inline void convert_channel(SrcInt src, DstInt& dst)
  {
    dst =
      to_rescaled_integral_channel<DstInt, double> (
      to_normalized_float_channel<SrcInt, double>(src) );
  }

  //! \brief Convert channels from a pixel vector to another pixel vector.
  template <typename T, typename U, int N>
  inline void convert_channel(const Matrix<T, N, 1>& src, Matrix<U, N, 1>& dst)
  {
    for (int i = 0; i < N; ++i)
      convert_channel(src[i], dst[i]);
  }

  //! \brief Convert channels from a pixel vector to another pixel vector.
  template <typename T, typename U, typename ColorSpace>
  inline void convert_channel(const Pixel<T, ColorSpace>& src,
                              Pixel<U, ColorSpace>& dst)
  {
    for (int i = 0; i < ColorSpace::size; ++i)
      convert_channel(src[i], dst[i]);
  }


} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_CORE_PIXEL_COLORSPACE_HPP */
