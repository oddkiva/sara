// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Shakti/MultiArray/MultiArray"

#include <boost/test/unit_test.hpp>

#include <DO/Shakti/Cuda/MultiArray.hpp>


namespace shakti = DO::Shakti;


BOOST_AUTO_TEST_CASE(test_constructor_1d)
{
  shakti::Array<float> array{10};
  BOOST_CHECK_EQUAL(10, array.sizes());
}

BOOST_AUTO_TEST_CASE(test_constructor_2d)
{
  shakti::MultiArray<float, 2> matrix{{3, 4}};
  BOOST_CHECK_EQUAL(shakti::Vector2i(3, 4), matrix.sizes());
  BOOST_CHECK_EQUAL(3, matrix.size(0));
  BOOST_CHECK_EQUAL(4, matrix.size(1));

  BOOST_CHECK_EQUAL(3, matrix.width());
  BOOST_CHECK_EQUAL(4, matrix.height());
}

BOOST_AUTO_TEST_CASE(test_constructor_3d)
{
  shakti::MultiArray<float, 3> matrix{{3, 4, 5}};
  BOOST_CHECK_EQUAL(shakti::Vector3i(3, 4, 5), matrix.sizes());
  BOOST_CHECK_EQUAL(3, matrix.size(0));
  BOOST_CHECK_EQUAL(4, matrix.size(1));
  BOOST_CHECK_EQUAL(5, matrix.size(2));

  BOOST_CHECK_EQUAL(3, matrix.width());
  BOOST_CHECK_EQUAL(4, matrix.height());
  BOOST_CHECK_EQUAL(5, matrix.depth());
}

BOOST_AUTO_TEST_CASE(test_copy_between_host_and_device_2d)
{
  const int w = 3;
  const int h = 4;
  // clang-format off
  float in_host_data[] = {
    0, 1, 2,
    3, 4, 5,
    6, 7, 8,
    9, 10, 11
  };
  // clang-format on

  // Copy to device.
  shakti::MultiArray<float, 2> out_device_image{in_host_data, {w, h}};

  // Copy back to host.
  float out_host_data[w * h];
  out_device_image.copy_to_host(out_host_data);

  BOOST_CHECK(std::equal(in_host_data, in_host_data + w * h, out_host_data));
}

BOOST_AUTO_TEST_CASE(test_copy_between_host_and_device_3d)
{
  const int w = 3;
  const int h = 4;
  const int d = 5;
  float in_host_data[w * h * d];
  for (int i = 0; i < w * h * d; ++i)
    in_host_data[i] = i;

  // Copy to device.
  shakti::MultiArray<float, 3> out_device_image{in_host_data, {w, h, d}};

  // Copy back to host.
  float out_host_data[w * h * d];
  out_device_image.copy_to_host(out_host_data);

  BOOST_CHECK(std::equal(in_host_data, in_host_data + w * h * d,  //
                         out_host_data));
}
