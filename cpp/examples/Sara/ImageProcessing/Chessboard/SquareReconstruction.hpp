#pragma once

#include <optional>
#include <unordered_set>

#include "Corner.hpp"


namespace DO::Sara {

  auto reconstruct_black_square_from_corner(
      const int c, const std::vector<Corner<float>>& corners,
      const std::vector<Eigen::Vector2f>& edge_grad_mean,
      const std::vector<Eigen::Matrix2f>& edge_grad_cov,
      const std::vector<std::unordered_set<int>>& edges_adjacent_to_corner,
      const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
      -> std::optional<std::array<int, 4>>;

  auto reconstruct_white_square_from_corner(
      const int c, const std::vector<Corner<float>>& corners,
      const std::vector<Eigen::Vector2f>& edge_grad_mean,
      const std::vector<Eigen::Matrix2f>& edge_grad_cov,
      const std::vector<std::unordered_set<int>>& edges_adjacent_to_corner,
      const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
      -> std::optional<std::array<int, 4>>;

}  // namespace DO::Sara
