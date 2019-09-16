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

#define BOOST_TEST_MODULE "ImageProcessing/Image Pyramid"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestImagePyramid)

BOOST_AUTO_TEST_CASE(test_image_pyramid_params)
{
  int first_octave_index = -1;
  int num_scales_per_octave = 3;
  double scale_geometric_factor = std::pow(2., 1. / num_scales_per_octave);
  int image_padding_size = 1;
  double scale_camera = 0.5;
  double scale_initial = 1.6;

  ImagePyramidParams pyramid_params(first_octave_index, num_scales_per_octave,
                                    scale_geometric_factor, image_padding_size,
                                    scale_camera, scale_initial);

  BOOST_CHECK_EQUAL(first_octave_index, pyramid_params.first_octave_index());
  BOOST_CHECK_EQUAL(num_scales_per_octave,
                    pyramid_params.num_scales_per_octave());
  BOOST_CHECK_EQUAL(scale_geometric_factor,
                    pyramid_params.scale_geometric_factor());
  BOOST_CHECK_EQUAL(image_padding_size, pyramid_params.image_padding_size());
  BOOST_CHECK_EQUAL(scale_camera, pyramid_params.scale_camera());
  BOOST_CHECK_EQUAL(scale_initial, pyramid_params.scale_initial());
}

BOOST_AUTO_TEST_CASE(test_image_pyramid)
{
  ImagePyramid<float> pyramid;

  int num_octaves = 2;
  int num_scales_per_octave = 3;
  float sigma_initial = 1.6f;
  float scale_geometric_factor = pow(2.f, 1 / 3.f);

  pyramid.reset(num_octaves, num_scales_per_octave, sigma_initial,
                scale_geometric_factor);

  BOOST_CHECK_EQUAL(num_octaves, pyramid.num_octaves());
  BOOST_CHECK_EQUAL(num_scales_per_octave, pyramid.num_scales_per_octave());
  BOOST_CHECK_EQUAL(sigma_initial, pyramid.scale_initial());
  BOOST_CHECK_EQUAL(scale_geometric_factor, pyramid.scale_geometric_factor());

  for (int i = 0; i < pyramid.num_octaves(); ++i)
    BOOST_CHECK_EQUAL(pyramid(i).size(),
                      static_cast<size_t>(pyramid.num_scales_per_octave()));
}

BOOST_AUTO_TEST_SUITE_END()
