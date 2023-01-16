// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Math/Rotation.hpp>

#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>


inline auto make_cube_vertices()
{
  auto cube = Eigen::MatrixXd{4, 8};
  // clang-format off
  cube.topRows(3) <<
    0, 1, 0, 1, 0, 1, 0, 1,
    0, 0, 1, 1, 0, 0, 1, 1,
    0, 0, 0, 0, 1, 1, 1, 1;
  // clang-format on
  cube.row(3).fill(1);

  // Recenter the cube.
  cube.topRows(3).colwise() += -0.5 * Eigen::Vector3d::Ones();

  return cube;
}

inline auto make_planar_chessboard_corners(int rows, int cols, double square_size)
{
  auto corners = Eigen::MatrixXd{4, rows * cols};
  for (auto y = 0; y < rows; ++y)
    for (auto x = 0; x < cols; ++x)
      corners.col(y * cols + x) << cols * square_size, rows * square_size, 0, 1;

  return corners;
}

inline auto make_relative_motion(double x = 0.1, double y = 0.3, double z = 0.2)
    -> DO::Sara::Motion
{
  using namespace DO::Sara;

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

inline auto make_camera(double x, double y, double z)
{
  const auto& [R, t] = make_relative_motion(x, y, z);
  return DO::Sara::normalized_camera(R, t);
}

inline auto to_camera_coordinates(const DO::Sara::PinholeCameraDecomposition& C,
                                  const Eigen::MatrixXd& X) -> Eigen::MatrixXd
{
  Eigen::MatrixXd X1 = (C.R * X.topRows(3)).colwise() + C.t;
  return X1.colwise().homogeneous();
}

inline auto project_to_film(const DO::Sara::PinholeCameraDecomposition& C,
                            const Eigen::MatrixXd& X) -> Eigen::MatrixXd
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
  return DO::Sara::TensorView_<T, 2>{const_cast<T*>(m.data()),
                                     {m.cols(), m.rows()}};
}
