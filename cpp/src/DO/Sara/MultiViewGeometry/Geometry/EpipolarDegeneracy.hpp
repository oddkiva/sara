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

#pragma once

#include <DO/Sara/MultiViewGeometry/Geometry/FundamentalMatrix.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/Homography.hpp>
#include <DO/Sara/MultiViewGeometry/Utilities.hpp>

#include <array>


namespace DO::Sara::degensac {

  // Two-view Geometry Estimation Unaffected by a Dominant Plane.
  //
  // Applicable for the essential matrix as well.
  //
  // We implement the Equation (4) of the paper.
  //
  // Points x1[i] and x2[i] satisfy the relation x2[i].T F * x1[i] = 0
  inline auto compute_homography(const FundamentalMatrix& F,             //
                                 const Eigen::Matrix<double, 2, 3>& x1,  //
                                 const Eigen::Matrix<double, 2, 3>& x2)  //
      -> Homography
  {
    const auto [e1, e2] = F.extract_epipoles();
    const Eigen::Matrix3d A = skew_symmetric_matrix(e2) * F.matrix();
    auto b = Eigen::Vector3d{};
    for (auto i = 0; i < 3; ++i)
    {
      const Eigen::Vector3d x1i = x1.col(i).homogeneous();
      const Eigen::Vector3d x2i = x2.col(i).homogeneous();
      const Eigen::Vector3d u = x2i.cross(A * x1i);
      const Eigen::Vector3d v = x2i.cross(e2);
      const auto normalization_factor = 1 / v.squaredNorm();
      b(i) = u.dot(v) * normalization_factor;
    }

    // clang-format off
    const Eigen::Matrix3d M_inverse = (Eigen::Matrix3d{} <<
      x1.col(0).homogeneous().transpose(),
      x1.col(1).homogeneous().transpose(),
      x1.col(2).homogeneous().transpose()
    ).finished().inverse();
    // clang-format on

    const Eigen::Matrix3d H = A - e2 * (M_inverse * b).transpose();

    return H;
  }

  // I have verified by counting manually.
  inline constexpr auto homography_index_triples()
      -> std::array<std::array<int, 3>, 5>
  {
    // 1-based enumeration from the paper.
    return {
        // {1, 2, 3}
        std::array{0, 1, 2},
        // {4, 5, 6}
        std::array{3, 4, 5},
        // {1, 2, 7}
        std::array{0, 1, 6},
        // {4, 5, 7}
        std::array{3, 4, 6},
        // {3, 6, 7}
        std::array{2, 5, 6}  //
    };
  }

}  // namespace DO::Sara::degensac
