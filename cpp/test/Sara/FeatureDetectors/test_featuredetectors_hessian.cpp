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

#define BOOST_TEST_MODULE "FeatureDetectors/Hessian Affine Detector"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/FeatureDetectors/Hessian.hpp>


using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestHessianAffineDetector)

BOOST_AUTO_TEST_CASE(test_hessian_laplace_detector)
{
  constexpr auto N = 2 * 10 + 1;
  auto I = Image<float>{N, N};
  I.flat_array().fill(0);
  I(1, 1) = 1.f;

  ComputeHessianLaplaceMaxima compute_hessian_laplace_maxima{};

  auto features = compute_hessian_laplace_maxima(I, 0);
}

BOOST_AUTO_TEST_CASE(test_determinant_of_hessian_detector)
{
  constexpr auto N = 2 * 10 + 1;
  auto I = Image<float>{N, N};
  I.flat_array().fill(0);
  I(1, 1) = 1.f;

  ComputeDoHExtrema compute_doh_maxima{};

  auto features = compute_doh_maxima(I, 0);
}

BOOST_AUTO_TEST_SUITE_END()
