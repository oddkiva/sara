// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015-2017 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "FeatureDescriptors/Dense Feature"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/FeatureDescriptors/DenseFeature.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestDenseFeature)

BOOST_AUTO_TEST_CASE(test_function)
{
  auto image = Image<float>{ 10, 10 };
  auto dense_sifts = compute_dense_sift(image);

  BOOST_CHECK_EQUAL(image.sizes(), dense_sifts.sizes());
  BOOST_CHECK_EQUAL(Vector128f::Zero(), dense_sifts(0, 0));
}

BOOST_AUTO_TEST_SUITE_END()
