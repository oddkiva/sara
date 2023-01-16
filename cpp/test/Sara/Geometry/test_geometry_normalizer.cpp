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

#define BOOST_TEST_MODULE "Geometry/Tools/Normalizer"

#include <DO/Sara/Geometry/Tools/Normalizer.hpp>

#include <boost/test/unit_test.hpp>


using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_compute_normalizer)
{
  auto X = Tensor_<float, 2>{3, 3};
  X.matrix() <<
    1, 1, 1,
    2, 2, 1,
    3, 3, 1;

  auto T = compute_normalizer(X);

  Matrix3f expected_T;
  expected_T <<
    0.5, 0.0, -0.5,
    0.0, 0.5, -0.5,
    0.0, 0.0,  1.0;

  BOOST_CHECK((T - expected_T).norm() < 1e-12);
}

BOOST_AUTO_TEST_CASE(test_apply_transform)
{
  auto X = Tensor_<double, 2>{3, 3};
  X.matrix() <<
    1, 1, 1,
    2, 2, 1,
    3, 3, 1;

  // From Oxford affine covariant features dataset, graf, H1to5P homography.
  auto H = Matrix3d{};
  H <<
    6.2544644e-01,  5.7759174e-02,  2.2201217e+02,
    2.2240536e-01,  1.1652147e+00, -2.5605611e+01,
    4.9212545e-04, -3.6542424e-05,  1.0000000e+00;

  auto HX = apply_transform(H, X);

  BOOST_CHECK(HX.matrix().col(2) == Vector3d::Ones());
}
