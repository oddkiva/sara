// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/TwoViewGeometry.hpp>


namespace DO::Sara {

  //! @addtogroup MultiviewErrorMeasures Error Measures
  //! @{

  struct SymmetricNormalizedEpipolarDistance
  {
    SymmetricNormalizedEpipolarDistance(const Eigen::Vector2i& image_sizes_1,
                               const Eigen::Vector2i& image_sizes_2)
      : image_sizes_1{image_sizes_1}
      , image_sizes_2{image_sizes_2}
      , norm_factor_1{normalization_factor(image_sizes_1)}
      , norm_factor_2{normalization_factor(image_sizes_2)}
    {
    }

    auto operator()(const Eigen::Matrix3d& F, const Eigen::Vector4d& pq) const
    {
      const auto p = pq.head(2).homogeneous();
      const auto q = pq.tail(2).homogeneous();
      // Algebraic square epipolar distance
      const double alg_epi_dist = q.transpose() * F * p;
      // Left line-point square distance
      const double right_dist = alg_epi_dist / (F * p).head(2).norm();
      // Right line-point square distance
      const double left_dist =
          alg_epi_dist / (F.transpose() * q).head(2).norm();
      const auto left_dist_normalized = norm_factor_1 * left_dist;
      const auto right_dist_normalized = norm_factor_2 * right_dist;
      return std::max(left_dist_normalized, right_dist_normalized);
    }

    static auto normalization_factor(const Eigen::Vector2i& sizes) -> double
    {
      const auto image_diagonal = sizes.cast<double>().norm();
      const auto image_area = static_cast<double>(sizes.x() * sizes.y());
      return 2 * image_diagonal / image_area;
    };

    const Eigen::Vector2i& image_sizes_1;
    const Eigen::Vector2i& image_sizes_2;
    const double norm_factor_1 = normalization_factor(image_sizes_1);
    const double norm_factor_2 = normalization_factor(image_sizes_2);
  };

  //! @}

} /* namespace DO::Sara */
