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

#include "LineReconstruction.hpp"
#include "SquareReconstruction.hpp"


namespace DO::Sara {

  static auto find_edge(
      const int a, const int b,
      const std::vector<std::unordered_set<int>>& edges_adjacent_to_corner,
      const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
      -> int
  {
    const auto& edges = edges_adjacent_to_corner[a];

    for (const auto& e : edges)
    {
      const auto corners = corners_adjacent_to_edge[e];
      if (corners.find(b) != corners.end() && corners.find(a) != corners.end())
        return e;
    }
    return -1;
  }

  // Greedy strategy...
  // Lots of parameters and dirty thresholds... see if we can do better.
  auto find_next_line_segment(
      const int ia, const int ib, const int current_edge_id,
      const std::vector<int>& edges_added,
      const std::vector<Corner<float>>& corners,
      const DO::Sara::CurveStatistics& edge_stats,
      const std::vector<Eigen::Vector2f>& edge_grad_mean,
      const std::vector<Eigen::Matrix2f>& edge_grad_cov,
      const std::vector<std::unordered_set<int>>& edges_adjacent_to_corner,
      const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
      -> std::pair<int, int>
  {
    const Eigen::Vector2f current_grad_mean =
        edge_grad_mean[current_edge_id].normalized();
    const auto& current_edge_inertia = edge_stats.inertias[current_edge_id];

    auto similarity = [](const Eigen::Matrix2d& a, const Eigen::Matrix2d& b) {
      return (a * b).trace() / (a.norm() * b.norm());
    };

    // Find the valid edges (those that are sufficiently good).
    auto valid_edges = std::vector<std::pair<int, float>>{};
    for (const auto& edge_id : edges_adjacent_to_corner[ib])
    {
      if (std::find(edges_added.begin(), edges_added.end(), edge_id) !=
          edges_added.end())
        continue;
      const Eigen::Vector2f ge = edge_grad_mean[edge_id].normalized();
      const auto gradient_oppositeness = current_grad_mean.dot(ge);
      if (gradient_oppositeness > cos(145.f / 180.f * M_PI))
        continue;

      const auto& edge_inertia = edge_stats.inertias[edge_id];
      const auto sim = similarity(current_edge_inertia, edge_inertia);
      if (sim < cos(45.f / 180.f * M_PI))
        continue;

      const auto& box = edge_stats.oriented_box(edge_id);
      const auto relative_thickness = box.lengths(1) / box.lengths(0);
      const auto is_thick = relative_thickness > 0.2 && box.lengths(1) > 5.;
      if (is_thick)
        continue;


      // The distribution of gradients should not show a broken edges.
      const auto& grad_cov = edge_grad_cov[edge_id];
      static constexpr auto kappa = 0.2f;
      using DO::Sara::square;
      const auto cornerness = grad_cov.determinant() -  //
                              kappa * square(grad_cov.trace());
      if (cornerness > 0)
        continue;

      valid_edges.emplace_back(edge_id, gradient_oppositeness);
    }

    if (valid_edges.empty())
      return {-1, -1};

    // Find the most collinear edge with opposite gradient direction.
    const auto best_edge_it = std::min_element(
        valid_edges.begin(), valid_edges.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    const auto& best_edge_id = best_edge_it->first;

    // Find the best vertex.
    const auto& a = corners[ia].coords;
    const auto& b = corners[ib].coords;
    const auto ab = (b - a).norm();

    auto best_ic = -1;
    auto best_ratio = 0.f;

    for (auto& ic : corners_adjacent_to_edge[best_edge_id])
    {
      if (ic == ia || ic == ib)
        continue;
      const auto& c = corners[ic].coords;
      const auto bc = (c - b).norm();
      const auto ratio = std::min(ab, bc) / std::max(ab, bc);
      if (ratio > 0.75 && ratio > best_ratio)
      {
        best_ic = ic;
        best_ratio = ratio;
      }
    }

    if (best_ic == -1)
      return {-1, -1};

    // The best line segment.
    return {ib, best_ic};
  };

  auto grow_line_from_square(
      const std::array<int, 4>& square, const int side,
      const std::vector<Corner<float>>& corners,
      const DO::Sara::CurveStatistics& edge_stats,
      const std::vector<Eigen::Vector2f>& edge_grad_mean,
      const std::vector<Eigen::Matrix2f>& edge_grad_cov,
      const std::vector<std::unordered_set<int>>& edges_adjacent_to_corner,
      const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
      -> std::vector<int>
  {
    auto ia = square[side];
    auto ib = square[(side + 1) % 4];
    auto e_ab =
        find_edge(ia, ib, edges_adjacent_to_corner, corners_adjacent_to_edge);
    if (e_ab == -1)
      return {ia, ib};

    auto line = std::vector<int>{};
    line.push_back(ia);
    line.push_back(ib);

    auto edges_added = std::vector<int>{};
    edges_added.push_back(e_ab);

    while (true)
    {
      std::tie(ia, ib) = find_next_line_segment(      //
          ia, ib, e_ab,                               //
          edges_added,                                //
          corners,                                    //
          edge_stats, edge_grad_mean, edge_grad_cov,  //
          edges_adjacent_to_corner, corners_adjacent_to_edge);
      if (ia == -1 || ib == -1)
        break;
      e_ab = find_edge(ia, ib,  //
                       edges_adjacent_to_corner, corners_adjacent_to_edge);

      line.push_back(ib);
      edges_added.push_back(e_ab);
    }


    ia = square[(side + 1) % 4];
    ib = square[side];
    e_ab =
        find_edge(ia, ib, edges_adjacent_to_corner, corners_adjacent_to_edge);
    edges_added.clear();
    edges_added.push_back(e_ab);

    auto line2 = std::vector<int>{};
    line2.push_back(ia);
    line2.push_back(ib);

    while (true)
    {
      std::tie(ia, ib) = find_next_line_segment(
          ia, ib, e_ab, edges_added, corners, edge_stats, edge_grad_mean,
          edge_grad_cov, edges_adjacent_to_corner, corners_adjacent_to_edge);
      if (ia == -1 || ib == -1)
        break;
      e_ab = find_edge(ia, ib,  //
                       edges_adjacent_to_corner, corners_adjacent_to_edge);

      line2.push_back(ib);
      edges_added.push_back(e_ab);
    }

    std::reverse(line2.begin(), line2.end());
    std::copy(line.begin() + 2, line.end(), std::back_inserter(line2));

    return line2;
  }


  auto grow_lines_from_square(
      const std::array<int, 4>& square,  //
      const std::vector<Corner<float>>& corners,
      const DO::Sara::CurveStatistics& edge_stats,
      const std::vector<Eigen::Vector2f>& edge_grad_mean,
      const std::vector<Eigen::Matrix2f>& edge_grad_cov,
      const std::vector<std::unordered_set<int>>& edges_adjacent_to_corner,
      const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
      -> std::vector<std::vector<int>>
  {
    auto lines = std::vector<std::vector<int>>{};
    for (auto side = 0; side < 4; ++side)
    {
      auto line = grow_line_from_square(square, side,  //
                                        corners,       //
                                        edge_stats, edge_grad_mean,
                                        edge_grad_cov, edges_adjacent_to_corner,
                                        corners_adjacent_to_edge);
      lines.emplace_back(std::move(line));
    }

    return lines;
  }

}  // namespace DO::Sara
