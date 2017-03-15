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

#define BOOST_TEST_MODULE "Core/Image/Image Class"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Pixel.hpp>
#include <DO/Sara/Core/Image/Image.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestImageClass)

BOOST_AUTO_TEST_CASE(test_2d_image_constructor)
{
  Image<int> image{ 10, 20 };
  BOOST_CHECK_EQUAL(image.width(), 10);
  BOOST_CHECK_EQUAL(image.height(), 20);

  Image<int, 3> volume{ 5, 10, 20 };
  BOOST_CHECK_EQUAL(volume.width(), 5);
  BOOST_CHECK_EQUAL(volume.height(), 10);
  BOOST_CHECK_EQUAL(volume.depth(), 20);

  Image<int, 3> volume2{ volume };
  BOOST_CHECK_EQUAL(volume2.width(), 5);
  BOOST_CHECK_EQUAL(volume2.height(), 10);
  BOOST_CHECK_EQUAL(volume2.depth(), 20);
}

BOOST_AUTO_TEST_CASE(test_matrix_view)
{
  auto A = Image<int>{ 2, 3 };
  A.matrix() <<
    1, 2,
    3, 4,
    5, 6;

  BOOST_CHECK_EQUAL(A(0, 0), 1); BOOST_CHECK_EQUAL(A(1, 0), 2);
  BOOST_CHECK_EQUAL(A(0, 1), 3); BOOST_CHECK_EQUAL(A(1, 1), 4);
  BOOST_CHECK_EQUAL(A(0, 2), 5); BOOST_CHECK_EQUAL(A(1, 2), 6);
}

BOOST_AUTO_TEST_SUITE_END()
