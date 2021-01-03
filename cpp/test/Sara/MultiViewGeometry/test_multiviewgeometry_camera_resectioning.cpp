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

#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Resectioning/HartleyZisserman.hpp>
#include <DO/Sara/MultiViewGeometry/Utilities.hpp>

#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;


auto make_cube_vertices()
{
  auto cube = Eigen::MatrixXd{4, 8};
  cube.topRows(3) <<
      // 0  1  2  3  4  5  6  7
      0,
      1, 0, 1, 0, 1, 0, 1,     //
      0, 0, 1, 1, 0, 0, 1, 1,  //
      0, 0, 0, 0, 1, 1, 1, 1;  //
  cube.row(3).fill(1);
  return cube;
}

auto make_relative_motion() -> sara::Motion
{
  using namespace sara;

  const Eigen::Matrix3d R = rotation_z(0.) * rotation_x(0.) * rotation_y(0.);
  // - The axes of the world coordinate system is has turned by the following
  //   rotational quantity.
  // - The columns of R are the vector coordinates of the world axes w.r.t.
  //   the camera coordinate system.

  const Eigen::Vector3d t{-2, -0.2, 10.};
  // - The vector t are the coordinates of the world center w.r.t. the camera
  //   coordinate system.

  return {R, t};
}

auto make_camera() -> sara::PinholeCamera
{
  const auto& [R, t] = make_relative_motion();
  return sara::normalized_camera(R, t);
}

auto project(const sara::PinholeCamera& C, const Eigen::MatrixXd& X)
    -> Eigen::MatrixXd
{
  auto x = Eigen::MatrixXd{3, X.cols()};
  x = C.matrix() * X;

  SARA_DEBUG << "x =\n" << x << std::endl;

  x = x.colwise().hnormalized();

  SARA_DEBUG << "x =\n" << x << std::endl;


  return x;
}


BOOST_AUTO_TEST_CASE(test_hartley_zisserman)
{
  auto X = make_cube_vertices();

  // Recenter the cube.
  X.topRows(3).colwise() += -0.5 * Eigen::Vector3d::Ones();

  // Translate the cube further 10 meters away from the first camera which will
  // considered to be the world coordinate frame.
  X.row(2).array() += 10;

  SARA_DEBUG << "3D vertices:" << std::endl;
  SARA_DEBUG << "X =\n" << X << std::endl;


  const auto C = make_camera();
  const auto x = project(C, X);
}
