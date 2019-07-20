// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018-2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "MultiViewGeometry/Relative Pose Estimator"

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/MultiViewGeometry/Estimators/RelativePoseEstimator.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Utilities.hpp>

#include <boost/test/unit_test.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestMultiViewGeometry)

auto generate_test_data()
{
  // 3D points.
  MatrixXd X(4, 5);  // coefficients are in [-1, 1].
  X.topRows<3>() <<
    -1.49998,   -0.5827,  -1.40591,  0.369386,  0.161931, //
    -1.23692, -0.434466, -0.142271, -0.732996,  -1.43086, //
     1.51121,  0.437918,   1.35859,   1.03883,  0.106923; //
  X.bottomRows<1>().fill(1.);

  const Matrix3d R = rotation_z(0.3) * rotation_x(0.1) * rotation_y(0.2);
  const Vector3d t{0.1, 0.2, 0.3};

  const auto E = essential_matrix(R, t);

  const Matrix34d C1 = PinholeCamera{Matrix3d::Identity(), Matrix3d::Identity(),
                                     Vector3d::Zero()};
  const Matrix34d C2 = PinholeCamera{Matrix3d::Identity(), R, t};
  MatrixXd x1 = C1 * X; x1.array().rowwise() /= x1.row(2).array();
  MatrixXd x2 = C2 * X; x2.array().rowwise() /= x2.row(2).array();

  return std::make_tuple(X, R, t, E, C1, C2, x1, x2);
}


BOOST_AUTO_TEST_CASE(test_relative_pose_estimator)
{
  const auto [X, R, t, E, C1, C2, u1, u2] = generate_test_data();

  const auto motions1 =
      RelativePoseEstimator<NisterFivePointAlgorithm>{}(u1, u2);

  const auto motions2 = RelativePoseEstimator<NisterFivePointAlgorithm>{
      CheiralityCriterion::NONE}(u1, u2);
  BOOST_CHECK_EQUAL(motions1.size(), motions2.size());

  const auto motions3 = RelativePoseEstimator<NisterFivePointAlgorithm>{
      CheiralityCriterion::CHEIRAL_COMPLETE}(u1, u2);
  BOOST_CHECK_LE(motions3.size(), motions2.size());

  const auto motions4 = RelativePoseEstimator<NisterFivePointAlgorithm>{
      CheiralityCriterion::CHEIRAL_MOST}(u1, u2);
  BOOST_CHECK_GE(motions4.size(), motions3.size());
}

BOOST_AUTO_TEST_SUITE_END()
