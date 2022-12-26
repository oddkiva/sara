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

  struct NormalizedEpipolarDistance
  {
    NormalizedEpipolarDistance(const Eigen::Vector2i& image_sizes_1,
                               const Eigen::Vector2i& image_sizes_2)
      : image_sizes_1{image_sizes_1}
      , image_sizes_2{image_sizes_2}
      , norm_factor_1{normalization_factor(image_sizes_1)}
      , norm_factor_2{normalization_factor(image_sizes_2)}
    {
    }

    auto operator()(const Eigen::Matrix3d& F, const Eigen::Vector4d& pq) const
    {
      const auto p = pq.head(2);
      const auto q = pq.tail(2);
      const double epipolar_distance =
          q.homogeneous().transpose() * F * p.homogeneous();
      const auto norm_dist_in_image_1 = norm_factor_1 * epipolar_distance;
      const auto norm_dist_in_image_2 = norm_factor_2 * epipolar_distance;
      return std::max(norm_dist_in_image_1, norm_dist_in_image_2);
    }

    static auto normalization_factor(const Eigen::Vector2i& sizes) -> double
    {
      const auto image_diagonal = sizes.cast<double>().array().square().sum();
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
