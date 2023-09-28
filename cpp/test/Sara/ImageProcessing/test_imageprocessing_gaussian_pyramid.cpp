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

BOOST_AUTO_TEST_CASE(test_gaussian_pyramid_with_fixed_octaves)
{
  auto I = Image<float>(16, 16);
  I.matrix().fill(1.f);

  const auto params = ImagePyramidParams{
      -1,    // first octave index
      2,     // 1 scale for the analysis + 1 scale for downsampling
      2.f,   // scale factor.
      1,     // image padding size
      0.5f,  // blur parameter in
      1.6f,  // scale_initial
      2      // max number of octaves.
  };
  BOOST_CHECK_EQUAL(params.num_octaves_max(), 2);

  const auto G = gaussian_pyramid(I, params);
  BOOST_CHECK_EQUAL(G.octave_count(), 2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_gaussian_pyramid, T, ChannelTypes)
{
  Image<T> I(16, 16);
  I.matrix().fill(PixelTraits<T>::max());

  const auto G = gaussian_pyramid(I, ImagePyramidParams(-1));
  const auto D = difference_of_gaussians_pyramid(G);
  const auto L = laplacian_pyramid(G);
}

BOOST_AUTO_TEST_SUITE_END()
