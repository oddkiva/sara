// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //
//
#define BOOST_TEST_MODULE "MultiViewGeometry/Epipolar Degeneracy"

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Math/Rotation.hpp>

#include <DO/Sara/MultiViewGeometry/Camera/PinholeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/EpipolarDegeneracy.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>

#include <boost/test/unit_test.hpp>

#include <iostream>


using namespace DO::Sara;
using namespace std;


struct TestData
{
  // 3D scene points.
  Eigen::MatrixXd X;

  // The camera matrices.
  Eigen::Matrix3d K1;
  Eigen::Matrix3d K2;

  // Two-view geometry.
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  EssentialMatrix E;
  FundamentalMatrix F;

  Matrix34d C1;
  Matrix34d C2;

  // Corresponding image points.
  Eigen::MatrixXd u1, u2;
};

auto generate_test_data() -> TestData
{
  // Pick 3D points on the plane z=5.
  Eigen::MatrixXd X(4, 5);
  // clang-format off
  X.topRows<3>() <<
    -1.,  1., 1., -1., 0.,
    -1., -1., 1.,  1., 0.,
     5.,  5., 5.,  5., 5.;
  // clang-format on
  X.bottomRows<1>().fill(1.);

  auto K1 = Eigen::Matrix3d{};
  auto K2 = Eigen::Matrix3d{};
  // clang-format off
  K1 <<
    1000,    0, 960,
       0, 1000, 540,
       0,    0,   1;

  K2 <<
    800,   0, 400,
      0, 800, 400,
      0,   0,   1;
  // clang-format on

  // Create a two-view geometry.
  const auto R = rotation(0.3, 0.2, 0.1);
  const auto t = Eigen::Vector3d{0.1, 0.2, 0.3};

  const auto E = essential_matrix(R, t);
  Eigen::Matrix3d F = K2.inverse().transpose() * E.matrix() * K1.inverse();
  F /= F.norm();

  const Matrix34d C1 =
      BasicPinholeCamera{K1, Matrix3d::Identity(), Vector3d::Zero()};
  const Matrix34d C2 = BasicPinholeCamera{K2, R, t};

  Eigen::MatrixXd x1 = (C1 * X).colwise().hnormalized();
  Eigen::MatrixXd x2 = (C2 * X).colwise().hnormalized();

  return {X, K1, K2, R, t, E, F, C1, C2, x1, x2};
}


BOOST_AUTO_TEST_CASE(check_homography_from_degenerate_epipolar_geometry)
{
  const auto [X, K1, K2, R, t, E, F, C1, C2, x1, x2] = generate_test_data();
  std::cout << "x1 =\n" << x1 << std::endl;
  std::cout << "x2 =\n" << x2 << std::endl;

  std::cout << "E =\n" << E << std::endl;
  std::cout << "F =\n" << F << std::endl;
  for (auto i = 0; i < x1.cols(); ++i)
    std::cout << "x2.T * F * x1 = "
              << x2.col(i).homogeneous().transpose() *
                     (F.matrix() * x1.col(i).homogeneous())
              << std::endl;

  auto H = degensac::compute_homography(F,  //
                                        x1.leftCols(3), x2.leftCols(3));
  H.matrix() /= H.matrix().norm();
  std::cout << "H =\n" << H << std::endl;

  for (auto i = 0; i < x1.cols(); ++i)
  {
    const Eigen::Vector2d Hx1 =
        (H.matrix() * x1.col(i).homogeneous()).hnormalized();
    std::cout << i << "  " << Hx1.transpose() << std::endl;
    std::cout << i << "  " << x2.col(i).transpose() << std::endl;

    BOOST_CHECK_SMALL((Hx1 - x2.col(i)).norm(), 1e-12);
  }
}
