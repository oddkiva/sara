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

#include <DO/Sara/ImageProcessing/InfiniteImage.hpp>

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

  const auto begin = Vector2i{-8, -8};
  const auto end = Vector2i{8, 8};
  auto dst = Image<float>{end-begin};

  PeriodicPadding pad;
  auto at = [&](const MultiArrayView<float, 2, ColMajor>& f, const Vector2i& x) -> float {
    return pad.at(f, x);
  };

  auto src_c = CoordsIterator<MultiArrayView<float, 2, ColMajor>>{begin, end};
  auto dst_i = dst.begin_array();
  for (; !dst_i.end(); ++src_c, ++dst_i)
  {
    std::cout << src_c->transpose() << "   " << at(src, *src_c) << std::endl;
    *dst_i = at(src, *src_c);
  }

  std::cout << std::endl;
  std::cout << "dst=\n" << dst.matrix() << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()
