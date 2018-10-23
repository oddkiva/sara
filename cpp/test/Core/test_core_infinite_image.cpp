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

#define BOOST_TEST_MODULE "ImageProcessing/Infinite Image"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/MultiArray/InfiniteMultiArrayView.hpp>
#include <DO/Sara/Core/Image.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestInfiniteImage)

BOOST_AUTO_TEST_CASE(test_infinite_image_with_periodic_padding)
{
  auto src = Image<float>{2, 2};
  src.matrix() <<
    0, 1,
    2, 3;

  const auto begin = Vector2i{-4, -4};
  const auto end = Vector2i{4, 4};
  const auto padding = PeriodicPadding{};

  const auto inf_src = make_infinite(src, padding);

  auto dst = Image<float>{end - begin};
  crop(dst, inf_src, begin, end);

  auto true_dst = Image<float>{end - begin};
  true_dst.matrix() <<
    0, 1, 1, 0, 0, 1, 1, 0,   // -4
    2, 3, 3, 2, 2, 3, 3, 2,   // -3
    2, 3, 3, 2, 2, 3, 3, 2,   // -2
    0, 1, 1, 0, 0, 1, 1, 0,   // -1
    0, 1, 1, 0, 0, 1, 1, 0,   // 0
    2, 3, 3, 2, 2, 3, 3, 2,   // 1
    2, 3, 3, 2, 2, 3, 3, 2,   // 2
    0, 1, 1, 0, 0, 1, 1, 0;   // 3

  BOOST_CHECK(true_dst.matrix() == dst.matrix());
}

BOOST_AUTO_TEST_CASE(test_infinite_image_with_constant_padding)
{
  auto src = Image<float>{2, 2};
  src.matrix() <<
    0, 1,
    2, 3;

  const auto begin = Vector2i{-2, -2};
  const auto end = Vector2i{4, 4};
  const auto padding = make_constant_padding(0.f);

  const auto inf_src = make_infinite(src, padding);

  auto dst = Image<float>{end - begin};
  crop(dst, inf_src, begin, end);

  auto true_dst = Image<float>{end - begin};
  true_dst.matrix() <<
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0,
    0, 0, 2, 3, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0;

  BOOST_CHECK(true_dst.matrix() == dst.matrix());
}

BOOST_AUTO_TEST_CASE(test_infinite_image_with_repeat_padding)
{
  auto src = Image<float>{2, 2};
  src.matrix() <<
    0, 1,
    2, 3;

  const auto begin = Vector2i{-2, -2};
  const auto end = Vector2i{4, 4};
  const auto padding = RepeatPadding{};

  const auto inf_src = make_infinite(src, padding);

  auto dst = Image<float>{end - begin};
  crop(dst, inf_src, begin, end);

  auto true_dst = Image<float>{end - begin};
  true_dst.matrix() <<
    0, 0, 0, 1, 1, 1,
    0, 0, 0, 1, 1, 1,
    0, 0, 0, 1, 1, 1,
    2, 2, 2, 3, 3, 3,
    2, 2, 2, 3, 3, 3,
    2, 2, 2, 3, 3, 3;

  BOOST_CHECK(true_dst.matrix() == dst.matrix());
}

BOOST_AUTO_TEST_CASE(test_infinite_image_with_periodic_padding_stepped_safe_crop)
{
  auto src = Image<float>{2, 2};
  src.matrix() <<
    0, 1,
    2, 3;

  const auto begin = Vector2i{0, 0};
  const auto end = Vector2i{10, 10};
  const auto steps = Vector2i{3, 3};
  const auto padding = PeriodicPadding{};

  const auto inf_src = make_infinite(src, padding);

  auto dst = Image<float>{end - begin};
  crop(dst, inf_src, begin, end);

  auto true_dst = Image<float>{4, 4};
  BOOST_CHECK(dst.sizes() == Vector2i(4, 4));
  true_dst.matrix() <<
    //0    1  2    3    4  5    6    7  8    9
      0, /*1, 1,*/ 0, /*0, 1,*/ 1, /*0, 0,*/ 1,  // 0
    //2, /*3, 3,*/ 2, /*2, 3,*/ 3, /*2, 2,*/ 3,  // 1
    //2, /*3, 3,*/ 2, /*2, 3,*/ 3, /*2, 2,*/ 3,  // 2
      0, /*1, 1,*/ 0, /*0, 1,*/ 1, /*0, 0,*/ 1,  // 3
    //0, /*1, 1,*/ 0, /*0, 1,*/ 1, /*0, 0,*/ 1,  // 4
    //2, /*3, 3,*/ 2, /*2, 3,*/ 3, /*2, 2,*/ 3,  // 5
      2, /*3, 3,*/ 2, /*2, 3,*/ 3, /*2, 2,*/ 3,  // 6
    //0, /*1, 1,*/ 0, /*0, 1,*/ 1, /*0, 0,*/ 1,  // 7
    //0, /*1, 1,*/ 0, /*0, 1,*/ 1, /*0, 0,*/ 1,  // 8
      2, /*3, 3,*/ 2, /*2, 3,*/ 3, /*2, 2,*/ 3;  // 9

  BOOST_CHECK(true_dst.matrix() == dst.matrix());
}

BOOST_AUTO_TEST_SUITE_END()
