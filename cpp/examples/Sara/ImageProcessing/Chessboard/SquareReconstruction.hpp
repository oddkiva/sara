#pragma once

#include <unordered_set>

#include "Corner.hpp"


enum class Direction : std::uint8_t
{
  Up = 0,
  Right = 1,
  Down = 2,
  Left = 3
};


auto direction_type(const float angle) -> Direction;

inline auto direction_type(const Eigen::Vector2f& d) -> Direction
{
  const auto angle = std::atan2(d.y(), d.x());
  return direction_type(angle);
}


auto reconstruct_black_square_from_corner(
    int c,                //
    int start_direction,  //
    const std::vector<Corner<float>>& corners,
    const std::vector<Eigen::Vector2f>& edge_grads,
    const std::vector<std::unordered_set<int>>& edges_adjacent_to_corner,
    const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
    -> std::optional<std::array<int, 4>>;

inline auto reconstruct_black_square_from_corner(
    const int c, const std::vector<Corner<float>>& corners,
    const std::vector<Eigen::Vector2f>& edge_grads,
    const std::vector<std::unordered_set<int>>& edges_adjacent_to_corner,
    const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
    -> std::optional<std::array<int, 4>>;
