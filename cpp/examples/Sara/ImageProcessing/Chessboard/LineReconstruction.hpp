#pragma once

#include <unordered_set>

#include "Corner.hpp"
#include "SquareReconstruction.hpp"


inline auto reconstruct_line(
    int c,                //
    int start_direction,  //
    const std::vector<Corner<float>>& corners,
    const std::vector<Eigen::Vector2f>& edge_grads,
    const std::vector<std::unordered_set<int>>& edges_adjacent_to_corner,
    const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
    -> std::vector<int>
{
  auto find_edge = [&](const std::unordered_set<int>& edges, Direction what) {
    for (const auto& e : edges)
      if (direction_type(edge_grads[e]) == what)
        return e;
    return -1;
  };

  auto find_next_vertex = [&](int edge, Direction what,
                              int current_vertex) -> int {
    struct Vertex
    {
      int id;
      float score;
      auto operator<(const Vertex& other) const
      {
        return score < other.score;
      }
    };

    const auto& cs = corners_adjacent_to_edge[edge];
    auto vs = std::vector<Vertex>(corners_adjacent_to_edge[edge].size());
    std::transform(cs.begin(), cs.end(), vs.begin(),
                   [&corners](const auto& v) -> Vertex {
                     return {v, corners[v].score};
                   });
    std::sort(vs.rbegin(), vs.rend());

    for (const auto& v : vs)
    {
      if (v.id == current_vertex)
        continue;

      const auto& a = corners[current_vertex].coords;
      const auto& b = corners[v.id].coords;
      const Eigen::Vector2f dir = (b - a).normalized();

      if (what == Direction::Up && dir.x() > 0)
        return v.id;
      if (what == Direction::Right && dir.y() > 0)
        return v.id;
      if (what == Direction::Down && dir.x() < 0)
        return v.id;
      if (what == Direction::Left && dir.y() < 0)
        return v.id;
    }
    return -1;
  };

  static const auto dirs = std::array{
      Direction::Up,
      Direction::Right,
      Direction::Down,
      Direction::Left,
  };


  std::vector<int> line;
  line.push_back(c);

  auto dir = dirs[start_direction];

  while (true)
  {
    const auto e = find_edge(edges_adjacent_to_corner[c], dir);
    if (e == -1)
      return line;
    const auto c = find_next_vertex(e, dir, line.back());
    if (c == -1)
      return line;
    line.push_back(c);
    dir = dirs[(start_direction + 2) % 4];
  }

  return line;
}
