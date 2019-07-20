// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "MultiViewGeometry/Homography Estimator"

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/MultiViewGeometry/Estimators/HomographyEstimator.hpp>
#include <DO/Sara/MultiViewGeometry/Estimators/ErrorMeasures.hpp>


#include <boost/test/unit_test.hpp>


using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_homography_estimators)
{
  auto u1 = MatrixXd(3, 4);
  auto u2 = MatrixXd(3, 4);

  u1 << 0, 1, 0, 1,
        0, 0, 1, 1,
        1, 1, 1, 1;

  u2 << 0.1, 1.2, 0.2, 2.0,
        0.1, 0.2, 1.4, 1.8,
        1.0, 1.0, 1.0, 1.0;

  auto hestimator = FourPointAlgorithm{};
  const auto [H] = hestimator(u1, u2);

  auto distance = SymmetricTransferError{H};
  const auto errors = distance(u1, u2);

  BOOST_CHECK_SMALL(errors.norm(), 1e-12);

  SARA_DEBUG << "Symmetric transfer errors =\n" << errors << std::endl;
}
