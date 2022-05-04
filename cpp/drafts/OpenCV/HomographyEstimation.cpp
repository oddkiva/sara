// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#include "OpenCVInterop.hpp"


namespace sara = DO::Sara;


auto estimate_H(const sara::OpenCV::Chessboard& chessboard) -> Eigen::Matrix3d
{
  const auto w = chessboard.width();
  const auto h = chessboard.height();
  const auto N = chessboard.corner_count();

  auto A = Eigen::MatrixXd{N * 2, 9};

  auto p1 = Eigen::MatrixXd{3, N};
  auto p2 = Eigen::MatrixXd{3, N};

  // Collect the 2D pixel coordinates.
  for (auto y = 0; y < h; ++y)
    for (auto x = 0; x < w; ++x)
      p1.col(y * w + x) = chessboard.image_point(x, y).homogeneous().cast<double>();

  // Keep it simple by just divide by 1000. Lazy but it works.
  //
  // clang-format off
  const auto T = (Eigen::Matrix3d{} <<
                  1e-3,    0, 0,
                     0, 1e-3, 0,
                     0,    0, 1).finished();
  // clang-format on
  const Eigen::Matrix3d invT = T.inverse();

  // Rescale the coordinates.
  p1 = T * p1;

  // Collect the 3D coordinates on the chessboard plane.
  for (auto y = 0; y < h; ++y)
    for (auto x = 0; x < w; ++x)
      p2.col(y * w + x) = chessboard.scene_point(x, y).homogeneous().cast<double>();

  // Form the data matrix used to determine H.
  for (auto i = 0; i < N; ++i)
  {
    // The image point
    const auto xi = p1.col(i);
    const auto ui = xi(0);
    const auto vi = xi(1);

    // The 3D coordinate on the chessboard plane.
    static const auto zero = Eigen::RowVector3d::Zero();
    const auto yiT = p2.col(i).transpose();

    A.row(2 * i + 0) << -yiT, zero, ui * yiT;
    A.row(2 * i + 1) << zero, -yiT, vi * yiT;
  }

  // SVD.
  const auto svd = A.jacobiSvd(Eigen::ComputeFullV);
  const auto V = svd.matrixV();
  const auto H_flat = V.col(8);
  SARA_DEBUG << "V =\n" << V << std::endl;
  SARA_DEBUG << "H_flat =\n" << H_flat.transpose() << std::endl;

  // clang-format off
  auto H = Eigen::Matrix3d{};
  H << H_flat.head(3).transpose(),
       H_flat.segment(3, 3).transpose(),
       H_flat.tail(3).transpose();
  // clang-format on
  SARA_DEBUG << "H =\n" << H << std::endl;

  H = invT * H;
  SARA_DEBUG << "H =\n" << H << std::endl;

  return H;
}
