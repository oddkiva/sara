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

#define BOOST_TEST_MODULE "Core/Pixel/Channel Conversions"

#include <cstdint>

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Pixel/ChannelConversion.hpp>
#include <DO/Sara/Core/Pixel/Pixel.hpp>


using namespace std;
using namespace DO::Sara;


// ========================================================================== //
// Define the set of integral channel types, which we will test.
using IntegralChannelTypes =
    boost::mpl::list<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t,
                     uint32_t, uint64_t>;

using IntegralChannelTypesRestricted =
    boost::mpl::list<int8_t, int16_t, uint8_t, uint16_t>;


// ========================================================================== //
// Test convert_channel function from integer type to floating-point type.
BOOST_AUTO_TEST_CASE_TEMPLATE(
    test_convert_channel_from_integer_type_to_floating_point_type, Int,
    IntegralChannelTypes)
{
  const Int channel_values[] = {numeric_limits<Int>::min(),
                                numeric_limits<Int>::max()};
  const float expected_float_values[] = {0, 1};
  const double expected_double_values[] = {0, 1};

  for (auto i = 0; i < 2; ++i)
  {
    // Using the explicit named function.
    {
      auto flt_val = to_normalized_float_channel<Int, float>(channel_values[i]);
      auto dbl_val =
          to_normalized_float_channel<Int, double>(channel_values[i]);
      BOOST_REQUIRE_EQUAL(flt_val, expected_float_values[i]);
      BOOST_REQUIRE_EQUAL(dbl_val, expected_double_values[i]);
    }

    // Using the unified API.
    {
      auto flt_val = float{};
      auto dbl_val = double{};
      convert_channel(channel_values[i], flt_val);
      convert_channel(channel_values[i], dbl_val);
      BOOST_REQUIRE_EQUAL(expected_float_values[i], flt_val);
      BOOST_REQUIRE_EQUAL(expected_double_values[i], dbl_val);
    }
  }
}


// ========================================================================== //
// Test channel conversion from floating-point type to integer type.
BOOST_AUTO_TEST_CASE_TEMPLATE(
    test_convert_channel_from_floating_point_type_to_integer_type, Int,
    IntegralChannelTypesRestricted)
{
  const auto eps = 1e-6f;
  const float channel_values[] = {eps, 1 - eps};

  const Int expected_int_values[] = {numeric_limits<Int>::min(),
                                     numeric_limits<Int>::max()};

  for (int i = 0; i < 2; ++i)
  {
    {
      auto val = to_rescaled_integral_channel<Int, float>(channel_values[i]);
      BOOST_REQUIRE_EQUAL(expected_int_values[i], val);
    }

    {
      auto val = to_rescaled_integral_channel<Int, double>(channel_values[i]);
      BOOST_REQUIRE_EQUAL(expected_int_values[i], val);
    }

    {
      auto val = Int{};
      convert_channel(channel_values[i], val);
      BOOST_REQUIRE_EQUAL(expected_int_values[i], val);
    }

    {
      auto val = Int{};
      convert_channel(static_cast<double>(channel_values[i]), val);
      BOOST_REQUIRE_EQUAL(expected_int_values[i], val);
    }
  }
}


// ========================================================================== //
// Test channel conversion between floating-point type.
BOOST_AUTO_TEST_CASE(test_convert_channel_between_float)
{
  for (int i = 0; i < 2; ++i)
  {
    auto double_converted_value = double{};
    convert_channel(float(i), double_converted_value);
    BOOST_REQUIRE_EQUAL(static_cast<double>(i), double_converted_value);

    auto float_converted_value = float{};
    convert_channel(double(i), float_converted_value);
    BOOST_REQUIRE_EQUAL(static_cast<float>(i), float_converted_value);
  }
}


// ========================================================================== //
// Test convert_channel function between integer types.
template <typename SrcInt, typename DstInt>
void test_channel_conversion_between_integer_types()
{
  const SrcInt test_src_values[] = {numeric_limits<SrcInt>::min(),
                                    numeric_limits<SrcInt>::max()};

  const DstInt expected_dst_values[] = {numeric_limits<DstInt>::min(),
                                        numeric_limits<DstInt>::max()};

  for (int i = 0; i < 2; ++i)
  {
    auto actual_dst_value = DstInt{};
    convert_channel(test_src_values[i], actual_dst_value);
    BOOST_REQUIRE_EQUAL(expected_dst_values[i], actual_dst_value);
  }
}

BOOST_AUTO_TEST_CASE(test_convert_channel_between_integers)
{
  test_channel_conversion_between_integer_types<int8_t, int16_t>();
  test_channel_conversion_between_integer_types<int8_t, int32_t>();
  test_channel_conversion_between_integer_types<int8_t, int64_t>();

  test_channel_conversion_between_integer_types<int8_t, uint8_t>();
  test_channel_conversion_between_integer_types<int8_t, uint16_t>();
  test_channel_conversion_between_integer_types<int8_t, uint32_t>();
  test_channel_conversion_between_integer_types<int8_t, uint64_t>();


  test_channel_conversion_between_integer_types<int16_t, int8_t>();
  test_channel_conversion_between_integer_types<int16_t, int32_t>();
  test_channel_conversion_between_integer_types<int16_t, int64_t>();

  test_channel_conversion_between_integer_types<int16_t, uint8_t>();
  test_channel_conversion_between_integer_types<int16_t, uint16_t>();
  test_channel_conversion_between_integer_types<int16_t, uint32_t>();
  test_channel_conversion_between_integer_types<int16_t, uint64_t>();


  test_channel_conversion_between_integer_types<int32_t, int8_t>();
  test_channel_conversion_between_integer_types<int32_t, int16_t>();
  test_channel_conversion_between_integer_types<int32_t, int64_t>();

  test_channel_conversion_between_integer_types<int32_t, uint8_t>();
  test_channel_conversion_between_integer_types<int32_t, uint16_t>();
  test_channel_conversion_between_integer_types<int32_t, uint32_t>();
  test_channel_conversion_between_integer_types<int32_t, uint64_t>();


  test_channel_conversion_between_integer_types<int64_t, int8_t>();
  test_channel_conversion_between_integer_types<int64_t, int16_t>();
  test_channel_conversion_between_integer_types<int64_t, int32_t>();

  test_channel_conversion_between_integer_types<int64_t, uint8_t>();
  test_channel_conversion_between_integer_types<int64_t, uint16_t>();
  test_channel_conversion_between_integer_types<int64_t, uint64_t>();
  test_channel_conversion_between_integer_types<int64_t, uint64_t>();


  test_channel_conversion_between_integer_types<uint8_t, int8_t>();
  test_channel_conversion_between_integer_types<uint8_t, int16_t>();
  test_channel_conversion_between_integer_types<uint8_t, int32_t>();
  test_channel_conversion_between_integer_types<uint8_t, int64_t>();

  test_channel_conversion_between_integer_types<uint8_t, uint16_t>();
  test_channel_conversion_between_integer_types<uint8_t, uint32_t>();
  test_channel_conversion_between_integer_types<uint8_t, uint64_t>();


  test_channel_conversion_between_integer_types<uint16_t, int8_t>();
  test_channel_conversion_between_integer_types<uint16_t, int16_t>();
  test_channel_conversion_between_integer_types<uint16_t, int32_t>();
  test_channel_conversion_between_integer_types<uint16_t, int64_t>();

  test_channel_conversion_between_integer_types<uint16_t, uint8_t>();
  test_channel_conversion_between_integer_types<uint16_t, uint32_t>();
  test_channel_conversion_between_integer_types<uint16_t, uint64_t>();


  test_channel_conversion_between_integer_types<uint32_t, int8_t>();
  test_channel_conversion_between_integer_types<uint32_t, int16_t>();
  test_channel_conversion_between_integer_types<uint32_t, int32_t>();
  test_channel_conversion_between_integer_types<uint32_t, int64_t>();

  test_channel_conversion_between_integer_types<uint32_t, uint8_t>();
  test_channel_conversion_between_integer_types<uint32_t, uint16_t>();
  test_channel_conversion_between_integer_types<uint32_t, uint64_t>();


  test_channel_conversion_between_integer_types<uint64_t, int8_t>();
  test_channel_conversion_between_integer_types<uint64_t, int16_t>();
  test_channel_conversion_between_integer_types<uint64_t, int32_t>();
  test_channel_conversion_between_integer_types<uint64_t, int64_t>();

  test_channel_conversion_between_integer_types<uint64_t, uint8_t>();
  test_channel_conversion_between_integer_types<uint64_t, uint16_t>();
  test_channel_conversion_between_integer_types<uint64_t, uint32_t>();
}
