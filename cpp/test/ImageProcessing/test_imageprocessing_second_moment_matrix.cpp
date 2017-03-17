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

#define BOOST_TEST_MODULE "ImageProcessing/Second Moment Matrix"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/ImageProcessing/SecondMomentMatrix.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestSecondMomentMatrix)

BOOST_AUTO_TEST_CASE(test_second_moment_matrix)
{
  auto vector_field = Image<Vector2f>{3, 3};
  vector_field.matrix().fill(Vector2f::Ones());

  auto true_moments = Image<Matrix2f>{3, 3};
  true_moments.flat_array().fill(Matrix2f::Ones());

  auto moments = vector_field.compute<SecondMomentMatrix>();
  for (size_t i = 0; i != moments.size(); ++i)
    BOOST_REQUIRE_SMALL_L2_DISTANCE(true_moments.flat_array()[i],
                                    moments.flat_array()[i], 1e-3f);
}

BOOST_AUTO_TEST_SUITE_END()
