#pragma once

#include <unordered_set>

#include <DO/Sara/ImageProcessing/EdgeShapeStatistics.hpp>

#include "SquareReconstruction.hpp"


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
