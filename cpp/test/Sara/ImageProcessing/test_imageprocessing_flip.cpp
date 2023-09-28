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

#define BOOST_TEST_MODULE "ImageProcessing/Image Flip"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/ImageProcessing/Flip.hpp>

#include "AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


class TestFixtureForImageFlip
{
protected:
  Image<int> image;

public:
  TestFixtureForImageFlip()
  {
    // Draw an 'F' letter.
    image.resize(4, 6);
    image.matrix() <<
      1, 1, 1, 1,
      1, 0, 0, 0,
      1, 1, 1, 0,
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 0, 0, 0;
  }
};

BOOST_FIXTURE_TEST_SUITE(TestImageFlip, TestFixtureForImageFlip)

BOOST_AUTO_TEST_CASE(test_flip_horizontally)
{
  auto true_flipped_image = Image<int>{4, 6};
  true_flipped_image.matrix() <<
      1, 1, 1, 1,
      0, 0, 0, 1,
      0, 1, 1, 1,
      0, 0, 0, 1,
      0, 0, 0, 1,
      0, 0, 0, 1;

  flip_horizontally(image);
  BOOST_REQUIRE_EQUAL(true_flipped_image.matrix(), image.matrix());
}

BOOST_AUTO_TEST_CASE(test_flip_vertically)
{
  auto true_flipped_image = Image<int>{4, 6};
  true_flipped_image.matrix() <<
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 1, 1, 0,
      1, 0, 0, 0,
      1, 1, 1, 1;

  flip_vertically(image);
  BOOST_REQUIRE_EQUAL(true_flipped_image.matrix(), image.matrix());
}

BOOST_AUTO_TEST_CASE(test_transpose)
{
  auto true_flipped_image = Image<int>{6, 4};
  true_flipped_image.matrix() <<
      1, 1, 1, 1, 1, 1,
      1, 0, 1, 0, 0, 0,
      1, 0, 1, 0, 0, 0,
      1, 0, 0, 0, 0, 0;

  transpose(image);
  BOOST_REQUIRE_EQUAL(true_flipped_image.matrix(), image.matrix());
}

BOOST_AUTO_TEST_CASE(test_transverse)
{
  auto true_flipped_image = Image<int>{6, 4};
  true_flipped_image.matrix() <<
      0, 0, 0, 0, 0, 1, 
      0, 0, 0, 1, 0, 1, 
      0, 0, 0, 1, 0, 1, 
      1, 1, 1, 1, 1, 1; 

  transverse(image);
  BOOST_REQUIRE_EQUAL(true_flipped_image.matrix(), image.matrix());
}

BOOST_AUTO_TEST_CASE(test_rotate_ccw_90)
{
  auto true_flipped_image = Image<int>{6, 4};
  true_flipped_image.matrix() <<
    1, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 0, 0,
    1, 1, 1, 1, 1, 1;

  rotate_ccw_90(image);
  BOOST_REQUIRE_EQUAL(true_flipped_image.matrix(), image.matrix());
}

BOOST_AUTO_TEST_CASE(test_rotate_ccw_180)
{
  auto true_flipped_image = Image<int>{4, 6};
  true_flipped_image.matrix() <<
      0, 0, 0, 1,
      0, 0, 0, 1,
      0, 0, 0, 1,
      0, 1, 1, 1,
      0, 0, 0, 1,
      1, 1, 1, 1;

  rotate_ccw_180(image);
  BOOST_REQUIRE_EQUAL(true_flipped_image.matrix(), image.matrix());
}

BOOST_AUTO_TEST_CASE(test_rotate_ccw_270)
{
  auto true_flipped_image = Image<int>{6, 4};
  true_flipped_image.matrix() <<
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 1, 0, 1,
    0, 0, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 1;

  rotate_ccw_270(image);
  BOOST_REQUIRE_EQUAL(true_flipped_image.matrix(), image.matrix());
}

BOOST_AUTO_TEST_CASE(test_rotate_cw_270)
{
  auto true_flipped_image = Image<int>{6, 4};
  true_flipped_image.matrix() <<
    1, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 0, 0,
    1, 1, 1, 1, 1, 1;

  rotate_cw_270(image);
  BOOST_REQUIRE_EQUAL(true_flipped_image.matrix(), image.matrix());
}

BOOST_AUTO_TEST_CASE(test_rotate_cw_180)
{
  auto true_flipped_image = Image<int>{4, 6};
  true_flipped_image.matrix() <<
      0, 0, 0, 1,
      0, 0, 0, 1,
      0, 0, 0, 1,
      0, 1, 1, 1,
      0, 0, 0, 1,
      1, 1, 1, 1;

  rotate_cw_180(image);
  BOOST_REQUIRE_EQUAL(true_flipped_image.matrix(), image.matrix());
}

BOOST_AUTO_TEST_CASE(test_rotate_cw_90)
{
  auto true_flipped_image = Image<int>{6, 4};
  true_flipped_image.matrix() <<
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 1, 0, 1,
    0, 0, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 1;

  rotate_cw_90(image);
  BOOST_REQUIRE_EQUAL(true_flipped_image.matrix(), image.matrix());
}

BOOST_AUTO_TEST_SUITE_END()
