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

#include <DO/Sara/ChessboardDetection/SquareReconstruction.hpp>
#include <DO/Sara/ImageProcessing/EdgeShapeStatistics.hpp>

#include <unordered_set>

namespace DO::Sara {

  auto grow_line_from_square(
      const std::array<int, 4>& square, const int side,
      const std::vector<Corner<float>>& corners,
      const CurveStatistics& edge_stats,
      const std::vector<Eigen::Vector2f>& edge_grad_mean,
      const std::vector<Eigen::Matrix2f>& edge_grad_cov,
      const std::vector<std::unordered_set<int>>& edges_adjacent_to_corner,
      const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
      -> std::vector<int>;

  auto grow_lines_from_square(
      const std::array<int, 4>& square,  //
      const std::vector<Corner<float>>& corners,
      const CurveStatistics& edge_stats,
      const std::vector<Eigen::Vector2f>& edge_grad_mean,
      const std::vector<Eigen::Matrix2f>& edge_grad_cov,
      const std::vector<std::unordered_set<int>>& edges_adjacent_to_corner,
      const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
      -> std::vector<std::vector<int>>;

}  // namespace DO::Sara
