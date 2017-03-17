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

#define BOOST_TEST_MODULE "ImageProcessing/Gaussian Pyramid"

#include <exception>

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>

#include <DO/Sara/ImageProcessing/GaussianPyramid.hpp>


using namespace std;
using namespace DO::Sara;


using ChannelTypes = boost::mpl::list<float, double, Rgb32f, Rgb64f>;

BOOST_AUTO_TEST_SUITE(TestGaussianPyramid)

BOOST_AUTO_TEST_CASE_TEMPLATE(test_gaussian_pyramid, T, ChannelTypes)
{
  Image<T> I(16, 16);
  I.matrix().fill(PixelTraits<T>::max());

  ImagePyramid<T> G(gaussian_pyramid(I, ImagePyramidParams(-1)));

  ImagePyramid<T> D(difference_of_gaussians_pyramid(G));

  ImagePyramid<T> L(laplacian_pyramid(G));
}

BOOST_AUTO_TEST_SUITE_END()
