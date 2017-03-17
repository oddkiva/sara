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

#define BOOST_TEST_MODULE "Core/Pixel/Pixel Class"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Pixel/ColorSpace.hpp>
#include <DO/Sara/Core/Pixel/Pixel.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestPixelClass)

BOOST_AUTO_TEST_CASE(test_rgb_32f)
{
  using Rgb32f = Pixel<float, Rgb>;

  Rgb32f red(1., 0, 0);
  BOOST_CHECK_EQUAL(red.channel<R>(), 1.f);
  BOOST_CHECK_EQUAL(red.channel<G>(), 0.f);
  BOOST_CHECK_EQUAL(red.channel<B>(), 0.f);
  BOOST_CHECK_EQUAL(red.num_channels(), 3);
}

BOOST_AUTO_TEST_SUITE_END()
