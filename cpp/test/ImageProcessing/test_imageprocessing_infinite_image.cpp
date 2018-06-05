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

BOOST_AUTO_TEST_CASE(test_upscale)
{
  auto src = Image<float>{2, 2};
  src.matrix() <<
    0, 1,
    2, 3;

  auto pad = PeriodicPadding{};
  auto inf_src = make_infinite(src, pad);

  const auto begin = Vector2i{-8, -8};
  const auto end = Vector2i{8, 8};

  auto src_i = inf_src.begin_subarray(begin, end);

  auto dst = Image<float>{end-begin};

  auto dst_i = dst.begin();
  for (; dst_i != dst.end(); ++src_i, ++dst_i)
    *dst_i = *src_i;

  std::cout << "dst=\n" << dst.matrix() << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()
