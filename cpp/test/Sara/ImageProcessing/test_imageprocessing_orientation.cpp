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

#define BOOST_TEST_MODULE "ImageProcessing/Orientation"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/ImageProcessing/Orientation.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestOrientation)

BOOST_AUTO_TEST_CASE(test_orientation)
{
  Image<Vector2f> vector_field(3, 3);
  vector_field.matrix().fill(Vector2f::Ones());

  Image<float> true_orientations(3, 3);
  true_orientations.flat_array().fill(static_cast<float>(M_PI_4));

  auto orientations = Image<float>{};
  BOOST_CHECK_THROW(orientation(vector_field, orientations), domain_error);

  orientations = Image<float>{vector_field.sizes()};
  orientation(vector_field, orientations);
  BOOST_CHECK_SMALL_L2_DISTANCE(true_orientations.matrix(),
                                orientations.matrix(), 1e-3f);

  orientations.clear();
  orientations = orientation(vector_field);
  BOOST_CHECK_SMALL_L2_DISTANCE(true_orientations.matrix(),
                                orientations.matrix(), 1e-3f);

  orientations.clear();
  orientations = vector_field.compute<Orientation>();
  BOOST_CHECK_SMALL_L2_DISTANCE(true_orientations.matrix(),
                                orientations.matrix(), 1e-3f);
}

BOOST_AUTO_TEST_SUITE_END()
