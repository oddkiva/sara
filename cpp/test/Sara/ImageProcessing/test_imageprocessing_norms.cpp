// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for.applyr
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "ImageProcessing/Norms"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/ImageProcessing/Norm.hpp>

#include "AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


class TestFixtureForNorms
{
protected:
  Image<Vector2f> vector_field;

public:
  TestFixtureForNorms()
  {
    vector_field.resize(3, 3);
    vector_field.matrix().fill(Vector2f::Ones());
  }
};

BOOST_FIXTURE_TEST_SUITE(TestNorms, TestFixtureForNorms)

BOOST_AUTO_TEST_CASE(test_squared_norm)
{
  auto true_squared_norm_image = Image<float>{3, 3};
  true_squared_norm_image.flat_array().fill(2);

  auto squared_norm_image = Image<float>{};
  BOOST_CHECK_THROW(squared_norm(vector_field, squared_norm_image),
                    domain_error);

  squared_norm_image.resize(vector_field.sizes());
  squared_norm(vector_field, squared_norm_image);
  BOOST_CHECK_EQUAL(true_squared_norm_image.matrix(),
                    squared_norm_image.matrix());

  squared_norm_image.clear();
  squared_norm_image = vector_field.compute<SquaredNorm>();
  BOOST_CHECK_EQUAL(true_squared_norm_image.matrix(),
                    squared_norm_image.matrix());
}

BOOST_AUTO_TEST_CASE(test_blue_norm)
{
  auto true_blue_norm_image = Image<float>{3, 3};
  true_blue_norm_image.flat_array().fill(Vector2f::Ones().blueNorm());

  auto blue_norm_image = Image<float>{};
  BOOST_CHECK_THROW(blue_norm(vector_field, blue_norm_image), domain_error);

  blue_norm_image.resize(vector_field.sizes());
  blue_norm(vector_field, blue_norm_image);
  BOOST_CHECK_EQUAL(true_blue_norm_image.matrix(), blue_norm_image.matrix());

  blue_norm_image = vector_field.compute<BlueNorm>();
  BOOST_CHECK_EQUAL(true_blue_norm_image.matrix(), blue_norm_image.matrix());
}

BOOST_AUTO_TEST_CASE(test_stable_norm)
{
  auto true_stable_norm_image = Image<float>{3, 3};
  true_stable_norm_image.flat_array().fill(Vector2f::Ones().stableNorm());

  auto stable_norm_image = Image<float>{};
  BOOST_CHECK_THROW(stable_norm(vector_field, stable_norm_image), domain_error);

  stable_norm_image.resize(vector_field.sizes());
  stable_norm(vector_field, stable_norm_image);
  BOOST_CHECK_EQUAL(true_stable_norm_image.matrix(),
                    stable_norm_image.matrix());

  stable_norm_image.clear();
  stable_norm_image = vector_field.compute<StableNorm>();
  BOOST_CHECK_EQUAL(true_stable_norm_image.matrix(),
                    stable_norm_image.matrix());
}

BOOST_AUTO_TEST_SUITE_END()
