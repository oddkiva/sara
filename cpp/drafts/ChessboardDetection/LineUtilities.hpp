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

#include "ChessboardDetector.hpp"


namespace DO::Sara {

  inline auto
  collect_lines(const ChessboardDetector::OrderedChessboardCorners& cb)
      -> std::vector<std::vector<Eigen::Vector2f>>
  {
    const auto m = cb.size();
    const auto n = cb.front().size();
    auto lines = std::vector<std::vector<Eigen::Vector2f>>{};

    for (auto i = 0u; i < m; ++i)
    {
      auto line = std::vector<Eigen::Vector2f>{};
      for (auto j = 0u; j < n; ++j)
        if (!std::isnan(cb[i][j].x()) && !std::isnan(cb[i][j].y()))
          line.push_back(cb[i][j]);

      lines.emplace_back(std::move(line));
    }

    for (auto j = 0u; j < n; ++j)
    {
      auto line = std::vector<Eigen::Vector2f>{};
      for (auto i = 0u; i < m; ++i)
        if (!std::isnan(cb[i][j].x()) && !std::isnan(cb[i][j].y()))
          line.push_back(cb[i][j]);

      lines.emplace_back(std::move(line));
    }
    return lines;
  }

  inline auto
  collect_lines(const ChessboardDetector::OrderedChessboardVertices& cb,
                const ChessboardDetector& detect)
      -> std::vector<std::vector<Eigen::Vector2f>>
  {
    const auto m = cb.size();
    const auto n = cb.front().size();
    auto lines = std::vector<std::vector<Eigen::Vector2f>>{};

    const auto add_line_points = [&detect](const int a, const int b,
                                           std::vector<Eigen::Vector2f>& line) {
      // Find the edges that connect the two vertices.
      const auto& edges_a = detect._edges_adjacent_to_corner.at(a);
      const auto& edges_b = detect._edges_adjacent_to_corner.at(b);

      auto edges_inter = std::unordered_set<int>{};
      for (const auto& e : edges_a)
        if (edges_b.find(e) != edges_b.end())
          edges_inter.insert(e);
      for (const auto& e : edges_b)
        if (edges_a.find(e) != edges_a.end())
          edges_inter.insert(e);

      if (edges_inter.empty())
        return;

      if (edges_inter.size() > 1)
        std::cerr << "Ambiguity!!!!" << std::endl;

      const auto edge_id = *edges_inter.begin();
      const auto& edge = detect._ed.pipeline.edges_as_list[edge_id];

      // Add the edge points.
      for (const auto& p : edge)
      {
        const Eigen::Vector2f pf =
            p.cast<float>() * detect._params.downscale_factor;
        line.push_back(pf);
      }
    };

    for (auto i = 0u; i < m; ++i)
    {
      auto line = std::vector<Eigen::Vector2f>{};
      for (auto j = 0u; j < n - 1; ++j)
      {
        const auto a = cb[i][j];
        const auto b = cb[i][j + 1];
        if (a == -1 || b == -1)
          continue;
        add_line_points(a, b, line);
      }

      lines.emplace_back(std::move(line));
    }

    for (auto j = 0u; j < n; ++j)
    {
      auto line = std::vector<Eigen::Vector2f>{};
      for (auto i = 0u; i < m - 1; ++i)
      {
        const auto a = cb[i][j];
        const auto b = cb[i + 1][j];
        if (a == -1 || b == -1)
          continue;
        add_line_points(a, b, line);
      }

      lines.emplace_back(std::move(line));
    }

    return lines;
  }

}  // namespace DO::Sara
