// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include "EdgeFusion.hpp"

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/FeatureDetectors/EdgePostProcessing.hpp>
#include <DO/Sara/FeatureDetectors/EdgeUtilities.hpp>
#include <DO/Sara/Graphics.hpp>


namespace DO::Sara {

  auto check_edge_grouping(                                          //
      const ImageView<Rgb8>& frame,                                  //
      const std::vector<Edge>& edges_refined,                        //
      const std::vector<std::vector<Eigen::Vector2i>>& edge_chains,  //
      const std::vector<Eigen::Vector2f>& mean_gradients,            //
      const std::vector<Eigen::Vector2d>& centers,                   //
      const std::vector<Eigen::Matrix2d>& axes,                      //
      const std::vector<Eigen::Vector2d>& lengths,                   //
      const Point2i& p1,                                             //
      double downscale_factor)                                       //
      -> void
  {
    tic();
    const auto edge_attributes = EdgeAttributes{.edges = edges_refined,
                                                .centers = centers,
                                                .axes = axes,
                                                .lengths = lengths};
    const auto edge_graph = EdgeGraph{edge_attributes, edge_chains, mean_gradients};
    toc("Edge Graph Initialization");

    tic();
    const auto edge_groups = edge_graph.group_by_alignment();
    toc("Edge Grouping By Alignment");

    tic();
    auto edges = std::vector<std::vector<Eigen::Vector2i>>{};
    edges.reserve(edges_refined.size());
    for (const auto& g : edge_groups)
    {
      auto edge_fused = std::vector<Eigen::Vector2i>{};
      for (const auto& e : g.second)
        append(edge_fused, edge_chains[e]);
      edges.push_back(std::move(edge_fused));
    }
    toc("Edge Fusion");

    // Display the quasi-straight edges.
    auto draw_task = [&]() {
      auto edge_group_colors = std::map<std::size_t, Rgb8>{};
      for (const auto& g : edge_groups)
        edge_group_colors[g.first] << rand() % 255, rand() % 255, rand() % 255;

      auto edge_colors = std::vector<Rgb8>(edges_refined.size(), Red8);
      for (const auto& g : edge_groups)
        for (const auto& e : g.second)
          edge_colors[e] = edge_group_colors[g.first];

      const Eigen::Vector2d p1d = p1.cast<double>();
      const auto& s = downscale_factor;

      auto detection = Image<Rgb8>{frame};
      detection.flat_array().fill(Black8);
      for (const auto& g : edge_groups)
      {
        for (const auto& e : g.second)
        {
          const auto& edge_refined = edges_refined[e];
          if (edge_refined.size() < 2)
            continue;

          const auto& color = edge_colors[e];
          draw_polyline(detection, edge_refined, color, p1d, s);

// #define DEBUG_SHAPE_STATISTICS
#ifdef DEBUG_SHAPE_STATISTICS
          const auto& rect = OrientedBox{.center = c,      //
                                         .axes = axes[e],  //
                                         .lengths = lengths[e]};
          rect.draw(detection, White8, p1d, s);
#endif
        }
      }
      display(detection);
      get_key();
    };

    tic();
    // draw_task();
    fill_rect(0, 0, frame.width(), frame.height(), Black8);
    for (const auto& edge : edges)
    {
      if (edge.size() < 2)
        continue;

      const auto color = Rgb8(rand() % 255, rand() % 255, rand() % 255);
      for (const auto& p : edge)
        fill_circle(p.x(), p.y(), 2, color);
    }
    toc("Draw");
  }

}  // namespace DO::Sara
