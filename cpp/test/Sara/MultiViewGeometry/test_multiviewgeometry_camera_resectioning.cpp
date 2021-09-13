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

#define BOOST_TEST_MODULE "MultiViewGeometry/Camera Resectioning"

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Math/Rotation.hpp>

#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Resectioning/HartleyZisserman.hpp>
#include <DO/Sara/MultiViewGeometry/Resectioning/LambdaTwist.hpp>
#include <DO/Sara/MultiViewGeometry/Utilities.hpp>

#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;


auto make_cube_vertices()
{
  auto cube = Eigen::MatrixXd{4, 8};
  cube.topRows(3) << 0, 1, 0, 1, 0, 1, 0, 1,  //
      0, 0, 1, 1, 0, 0, 1, 1,                 //
      0, 0, 0, 0, 1, 1, 1, 1;                 //
  cube.row(3).fill(1);

  // Recenter the cube.
  cube.topRows(3).colwise() += -0.5 * Eigen::Vector3d::Ones();

  return cube;
}

auto make_relative_motion(double x = 0.1, double y = 0.3, double z = 0.2)
    -> sara::Motion
{
  using namespace sara;

  // Euler composite rotation.
  const Eigen::Matrix3d R = rotation(z, y, x);
  // - The axes of the world coordinate system has turned by the following
  //   rotational quantity.
  // - The columns of R are the vector coordinates of the world axes w.r.t.
  //   the camera coordinate system.

  const Eigen::Vector3d t{-2, -0.2, 10.};
  // - The vector t are the coordinates of the world center w.r.t. the camera
  //   coordinate system.

  return {R, t};
}

auto make_camera(double x, double y, double z) -> sara::PinholeCamera
{
  const auto& [R, t] = make_relative_motion(x, y, z);
  return sara::normalized_camera(R, t);
}

auto to_camera_coordinates(const sara::PinholeCamera& C,
                           const Eigen::MatrixXd& X) -> Eigen::MatrixXd
{
  Eigen::MatrixXd X1 = (C.R * X.topRows(3)).colwise() + C.t;
  return X1.colwise().homogeneous();
}

auto project_to_film(const sara::PinholeCamera& C, const Eigen::MatrixXd& X)
    -> Eigen::MatrixXd
{
  auto xh = Eigen::MatrixXd{3, X.cols()};
  xh = C.matrix() * X;

  auto x = Eigen::MatrixXd{2, X.cols()};
  x = xh.colwise().hnormalized();

  return x;
}

template <typename T, int M, int N>
inline auto tensor_view(const Eigen::Matrix<T, M, N>& m)
{
  return sara::TensorView_<T, 2>{const_cast<T*>(m.data()),
                                 {m.cols(), m.rows()}};
}


BOOST_AUTO_TEST_CASE(test_flipud)
{
  auto A = Eigen::Matrix3i{};
  A << 1, 2, 3,  //
      4, 5, 6,   //
      7, 8, 9;

  const auto A_flipped = sara::flipud(A);
  auto A_flipped_true = Eigen::Matrix3i{};
  A_flipped_true << 7, 8, 9,  //
      4, 5, 6,                //
      1, 2, 3;

  BOOST_CHECK(A_flipped_true == A_flipped);
}

BOOST_AUTO_TEST_CASE(test_fliplr)
{
  auto A = Eigen::Matrix3i{};
  A << 1, 2, 3,  //
      4, 5, 6,   //
      7, 8, 9;

  const auto A_flipped = sara::fliplr(A);
  auto A_flipped_true = Eigen::Matrix3i{};
  A_flipped_true << 3, 2, 1,  //
      6, 5, 4,                //
      9, 8, 7;
  BOOST_CHECK(A_flipped_true == A_flipped);
}

BOOST_AUTO_TEST_CASE(test_hartley_zisserman)
{
  const auto xa = std::array{0.0, 0.1, 0.3, 0.0};
  const auto ya = std::array{0.0, 0.2, 0.2, 0.1};
  const auto za = std::array{0.0, 0.3, 0.1, 0.0};
  auto Xw = make_cube_vertices();


  auto check = [&](int i) {
    // Translate the cube further 10 meters away from the world center.
    Xw.row(2).array() += 10;

    std::cout << "* World Coordinates:" << std::endl;
    std::cout << "  Xw =\n" << Xw << std::endl;

    // Get the test camera matrix.
    const auto C = make_camera(xa[i], ya[i], za[i]);

    auto Xc = to_camera_coordinates(C, Xw);
    std::cout << "* Camera Coordinates:" << std::endl;
    std::cout << "  Xc =\n" << Xc << std::endl;

    auto x = project_to_film(C, Xw);
    std::cout << "* Film Coordinates:" << std::endl;
    std::cout << "  x =\n" << x << std::endl;


    // Now check the resectioning method.
    const auto Xw1 = tensor_view(Xw);
    const auto x1 = tensor_view(x);
    const auto [K, R, t] = sara::resectioning_hartley_zisserman(Xw1, x1);
    const auto C1 = sara::PinholeCamera{K, R, t};


    std::cout << "Calibration matrix:" << std::endl;
    std::cout << "K1 =\n" << K << std::endl;
    std::cout << std::endl;

    std::cout << "Rotations:" << std::endl;
    std::cout << "R =\n" << C.R << std::endl;
    std::cout << "R1 =\n" << R << std::endl;
    std::cout << std::endl;

    std::cout << "Translations:" << std::endl;
    std::cout << "t =\n" << C.t << std::endl;
    std::cout << "t1 =\n" << t << std::endl;
    std::cout << std::endl;

    std::cout << "Algebraic Difference:" << std::endl;
    std::cout << "C1 - C =\n" << C1.matrix() - C.matrix() << std::endl;
    std::cout << std::endl;

    BOOST_REQUIRE_LE((C1.matrix() - C.matrix()).norm(), 1e-8);
  };

  for (auto i = 0u; i < xa.size(); ++i)
    check(i);
}


BOOST_AUTO_TEST_CASE(test_lambda_twist)
{
  const auto xa = std::array{0.0, 0.1, 0.3, 0.0};
  const auto ya = std::array{0.0, 0.2, 0.2, 0.1};
  const auto za = std::array{0.0, 0.3, 0.1, 0.0};
  auto Xw = make_cube_vertices();

  // Get the test camera matrix.
  for (auto i = 0; i < 1; ++i)
  {
    const auto C = make_camera(xa[i], ya[i], za[i]);

    auto Xc = to_camera_coordinates(C, Xw);
    std::cout << "* Camera Coordinates:" << std::endl;
    std::cout << "  Xc =\n" << Xc << std::endl;

    auto Yc = Xc.topRows<3>().colwise().normalized();
    std::cout << "* Backprojected Light Rays:" << std::endl;
    std::cout << "  Yc =\n" << Yc << std::endl;
    std::cout << "* Yc column norms " << std::endl;
    std::cout << "  column_norm(Yc) = " << Yc.colwise().norm() << std::endl;

    auto lambda_twist =
        sara::LambdaTwist<double>{Xw.topLeftCorner<3, 3>(), Yc.leftCols<3>()};

    auto E = Eigen::Matrix3d{};
    auto theta = static_cast<double>(M_PI) / 6;
    // clang-format off
    E <<
      std::cos(theta), -std::sin(theta), 0,
      std::sin(theta),  std::cos(theta), 0,
                    0,                0, 1;
    // clang-format on
    SARA_DEBUG << "E**2 =\n" << E.array().square() << std::endl;

    auto S = Eigen::Vector3d{1.1, -0.6, 0};

    const Eigen::Matrix3d M = E * S.asDiagonal() * E.transpose();
    SARA_DEBUG << "M = \n" << M << std::endl;

    auto E1 = Eigen::Matrix3d{};
    auto S1 = Eigen::Vector3d{};

// #define EIGEN_IMPL
#if defined(EIGEN_IMPL)
    // More robust, much simpler and also direct.
    auto eigenSolver = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>{};
    eigenSolver.computeDirect(M);
    std::cout << "Eigenvalues = " << eigenSolver.eigenvalues().transpose()
              << std::endl;
    std::cout << "Eigenvectors = " << std::endl
              << eigenSolver.eigenvectors() << std::endl;

    // The first eigenvalue is always negative, the second is zero, and the
    // third one is positive.
    // The right-handedness is preserved if we rotate the column-vectors.
    E1.col(0) = eigenSolver.eigenvectors().col(2);
    E1.col(1) = eigenSolver.eigenvectors().col(0);
    E1.col(2) = eigenSolver.eigenvectors().col(1);

    S1(0) = eigenSolver.eigenvalues()(2);
    S1(1) = eigenSolver.eigenvalues()(0);
    S1(2) = eigenSolver.eigenvalues()(1);

    // Yes it is a bit slower, but this should be OK.
#else  // MINE
    lambda_twist.eig3x3known0(M, E1, S1);
#endif

    SARA_DEBUG << "E1 = \n" << E1 << std::endl;
    SARA_DEBUG << "S1 = " << S1.transpose() << std::endl;
  }
}
