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

#pragma once

#include <DO/Sara/ImageProcessing/EdgeShapeStatistics.hpp>


namespace DO::Sara {

  auto get_curve_shape_statistics(
      const std::vector<std::vector<Eigen::Vector2i>>& curve_pts)
      -> CurveStatistics;

  auto
  gradient_mean(const std::vector<std::vector<Eigen::Vector2i>>& curve_pts,  //
                const ImageView<float>& Ix,                                  //
                const ImageView<float>& Iy)                                  //
      -> std::vector<Eigen::Vector2f>;

  auto gradient_covariance(
      const std::vector<std::vector<Eigen::Vector2i>>& curve_pts,  //
      const ImageView<float>& Ix,                                  //
      const ImageView<float>& Iy)                                  //
      -> std::vector<Eigen::Matrix2f>;

}  // namespace DO::Sara
