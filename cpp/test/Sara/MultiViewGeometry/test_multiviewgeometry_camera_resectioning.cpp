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

#include "SyntheticDataUtilities.hpp"

#include <DO/Sara/MultiViewGeometry/Resectioning/HartleyZisserman.hpp>

#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;


BOOST_AUTO_TEST_CASE(test_flipud)
{
  auto A = Eigen::Matrix3i{};
  A << 1, 2, 3,  //
      4, 5, 6,   //
      7, 8, 9;

  const auto A_flipped = sara::flipud(A);
  auto A_flipped_true = Eigen::Matrix3i{};
  // clang-format off
  A_flipped_true <<
      7, 8, 9,
      4, 5, 6,
      1, 2, 3;
  // clang-format on

  BOOST_CHECK(A_flipped_true == A_flipped);
}

BOOST_AUTO_TEST_CASE(test_fliplr)
{
  auto A = Eigen::Matrix3i{};
  // clang-format off
  A <<
    1, 2, 3,
    4, 5, 6,
    7, 8, 9;
  // clang-format on

  const auto A_flipped = sara::fliplr(A);
  auto A_flipped_true = Eigen::Matrix3i{};
  // clang-format off
  A_flipped_true <<
    3, 2, 1,
    6, 5, 4,
    9, 8, 7;
  // clang-format on
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
    SARA_CHECK(Xw1.sizes().transpose());
    SARA_CHECK(x1.sizes().transpose());
    const auto [K, R, t] = sara::resectioning_hartley_zisserman(Xw1, x1);
    const auto C1 = sara::PinholeCameraDecomposition{K, R, t};

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
