// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2023-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "MultipleObjectTracking/Kalman Filter Equations"

#include <DO/Sara/MultipleObjectTracking/ObservationModel.hpp>
#include <DO/Sara/MultipleObjectTracking/StateTransitionModel.hpp>

#include <boost/test/unit_test.hpp>


namespace mot = DO::Sara::MultipleObjectTracking;


BOOST_AUTO_TEST_SUITE(TestKalmanFilter)

BOOST_AUTO_TEST_CASE(test_observation_equation)
{
  auto z = mot::ObservationDistribution<float>{};
  z.μ << 1, 1, 0.5, 1.75;

  auto x = mot::StateDistribution<float>{};
  x.μ.setZero();
  x.μ.head<4>() << z.μ;
  x.Σ.setZero();

  const auto obs_eq = mot::ObservationEquation<float>{};

  const auto r = obs_eq.residual(z, x);
  BOOST_CHECK_SMALL(r.norm(), 1e-6f);
}

BOOST_AUTO_TEST_SUITE_END()
