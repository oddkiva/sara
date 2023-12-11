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

#define BOOST_TEST_MODULE "Kalman Filter/Concepts"

#include <DO/Sara/KalmanFilter/DistributionConcepts.hpp>
#include <DO/Sara/KalmanFilter/MatrixConcepts.hpp>
#include <DO/Sara/MultipleObjectTracking/BaseDefinitions.hpp>

#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;
namespace mot = DO::Sara::MultipleObjectTracking;


BOOST_AUTO_TEST_SUITE(TestKalmanFilter)

BOOST_AUTO_TEST_CASE(test_matrix_concepts)
{
  [[maybe_unused]] const sara::VectorConcept auto x = Eigen::Vector2d{};
  [[maybe_unused]] const sara::CompileTimeSquareMatrixConcept auto m =
      Eigen::Matrix4f{};
}

BOOST_AUTO_TEST_CASE(test_distribution_concepts)
{
  [[maybe_unused]] const sara::GaussianDistribution auto x =
      mot::StateDistribution<double>{};

  [[maybe_unused]] const sara::GaussianDistribution auto z =
      mot::ObservationDistribution<double>{};
}


BOOST_AUTO_TEST_SUITE_END()
