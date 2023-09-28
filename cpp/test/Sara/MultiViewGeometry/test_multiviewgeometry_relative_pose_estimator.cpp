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

#include <DO/Sara/Core/Math/Rotation.hpp>
#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/RelativePoseSolver.hpp>

#include <boost/test/unit_test.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>


using namespace std;
using namespace DO::Sara;

struct TestData
{
  MatrixXd X;

  Matrix3d R;
  Vector3d t;
  EssentialMatrix E;

  Matrix34d C1;
  Matrix34d C2;

  MatrixXd u1, u2;
};

auto generate_test_data() -> TestData
{
  // 3D points.
  MatrixXd X(4, 5);  // coefficients are in [-1, 1].
  // clang-format off
  X.topRows<3>() <<
    -1.49998,   -0.5827,  -1.40591,  0.369386,  0.161931, //
    -1.23692, -0.434466, -0.142271, -0.732996,  -1.43086, //
     1.51121,  0.437918,   1.35859,   1.03883,  0.106923; //
  // clang-format on
  X.bottomRows<1>().fill(1.);

  const Matrix3d R = rotation(0.3, 0.2, 0.1);
  const Vector3d t{.1, 0.2, 0.3};

  const auto E = essential_matrix(R, t);

  const Matrix34d C1 = PinholeCameraDecomposition{
      Matrix3d::Identity(), Matrix3d::Identity(), Vector3d::Zero()};
  const Matrix34d C2 = PinholeCameraDecomposition{Matrix3d::Identity(), R, t};
  MatrixXd x1 = C1 * X;
  x1.array().rowwise() /= x1.row(2).array();
  MatrixXd x2 = C2 * X;
  x2.array().rowwise() /= x2.row(2).array();

  return {X, R, t, E, C1, C2, x1, x2};
}

BOOST_AUTO_TEST_SUITE(TestMultiViewGeometry)


BOOST_AUTO_TEST_CASE(test_relative_pose_estimator)
{
  // const auto [X, R, t, E, C1, C2, u1, u2] = generate_test_data();
  const auto test_data = generate_test_data();
  const auto& u1 = test_data.u1;
  const auto& u2 = test_data.u2;

  const auto motions1 = RelativePoseSolver<NisterFivePointAlgorithm>{}(u1, u2);

  const auto motions2 = RelativePoseSolver<NisterFivePointAlgorithm>{}(u1, u2);
  BOOST_CHECK_EQUAL(motions1.size(), motions2.size());

  const auto motions3 = RelativePoseSolver<NisterFivePointAlgorithm>{}(u1, u2);
  BOOST_CHECK_LE(motions3.size(), motions2.size());

  const auto motions4 = RelativePoseSolver<NisterFivePointAlgorithm>{}(u1, u2);
  BOOST_CHECK_GE(motions4.size(), motions3.size());
}

BOOST_AUTO_TEST_SUITE_END()
