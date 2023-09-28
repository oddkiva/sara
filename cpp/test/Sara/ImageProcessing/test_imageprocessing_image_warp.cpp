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

#define BOOST_TEST_MODULE "ImageProcessing/Image Warp"

#include <exception>

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Pixel/Typedefs.hpp>
#include <DO/Sara/ImageProcessing/Warp.hpp>

#include "AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


using ChannelTypes = boost::mpl::list<float, double>;

BOOST_AUTO_TEST_SUITE(TestImageWarp)

BOOST_AUTO_TEST_CASE_TEMPLATE(test_image_warp, T, ChannelTypes)
{
  Image<T> src(3, 3);
  src.matrix() <<
    0, 1, 2,
    3, 4, 5,
    6, 7, 8;

  Matrix3d homography;
  homography <<
    1, 0, 0,
    0, 1, 0,
    0, 0, 1;

  Image<T> dst(3, 3);
  warp(src, dst, homography);
  BOOST_CHECK_CLOSE_L2_DISTANCE(src.matrix(), dst.matrix(),
                                static_cast<T>(1e-7));
}

BOOST_AUTO_TEST_SUITE_END()
