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

#define BOOST_TEST_MODULE "ImageProcessing/Determinant"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/ImageProcessing/Determinant.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestDeterminant)

BOOST_AUTO_TEST_CASE(test_determinant)
{
  auto tensor = Image<Matrix2f>{3, 3};
  tensor.flat_array().fill(Matrix2f::Ones());

  auto det = tensor.compute<Determinant>();

  for (int i = 0; i < det.flat_array().size(); ++i)
    BOOST_REQUIRE_CLOSE(0, det.flat_array()[i], 1e-5);
}

BOOST_AUTO_TEST_SUITE_END()
