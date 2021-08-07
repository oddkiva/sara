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
#include "Otsu.hpp"

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/FeatureDetectors/EdgePostProcessing.hpp>
#include <DO/Sara/FeatureDetectors/EdgeUtilities.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>


namespace DO::Sara {

  auto mean_colors(const std::map<int, std::vector<Eigen::Vector2i>>& regions,
                   const Image<Rgb8>& image)
  {
    auto colors = std::map<int, Rgb8>{};
    for (const auto& [label, points] : regions)
    {
      const auto num_points = static_cast<float>(points.size());
      Eigen::Vector3f color = Vector3f::Zero();
      for (const auto& p : points)
        color += image(p).cast<float>();
      color /= num_points;

      colors[label] = color.cast<std::uint8_t>();
    }
    return colors;
  }

  auto check_edge_grouping(                                          //
      const ImageView<Rgb8>& frame,                                  //
      const std::vector<Edge<double>>& edges_refined,                //
      const std::vector<Edge<int>>& edge_chains,                     //
      const std::vector<Eigen::Vector2f>& mean_gradients,            //
      const std::vector<Eigen::Vector2d>& centers,                   //
      const std::vector<Eigen::Matrix2d>& axes,                      //
      const std::vector<Eigen::Vector2d>& lengths)                   //
      -> void
  {
    tic();
    const auto edge_attributes =
        EdgeAttributes{edges_refined, centers, axes, lengths};
    const auto edge_graph = EdgeGraph{edge_attributes, edge_chains, mean_gradients};
    toc("Edge Graph Initialization");

    tic();
    const auto edge_groups = edge_graph.group_by_alignment();
    toc("Edge Grouping By Alignment");

    tic();
    auto edges = std::vector<std::vector<Eigen::Vector2i>>{};
    auto edge_gradients = std::vector<Eigen::Vector2f>{};
    edges.reserve(edges_refined.size());
    edge_gradients.reserve(edges_refined.size());
    for (const auto& g : edge_groups)
    {
      auto edge_fused = std::vector<Eigen::Vector2i>{};

      // Fuse the edges.
      for (const auto& e : g.second)
        append(edge_fused, edge_chains[e]);

      // Calculate the cardinality of the fused edge.
      const auto fused_edge_cardinality =
          std::accumulate(g.second.begin(), g.second.end(), 0,
                          [&](const auto& a, const auto& b) {
                            return a + static_cast<int>(edge_chains[b].size());
                          });

      // Calculate the gradient of the fused edge.
      const Eigen::Vector2f fused_edge_gradient = std::accumulate(
          g.second.begin(), g.second.end(), Eigen::Vector2f{0, 0},
          [&](const auto& a, const auto& b) -> Eigen::Vector2f {
            return a + edge_chains[b].size() * mean_gradients[b];
          }) / fused_edge_cardinality;

      // Store the fused edge.
      edges.push_back(std::move(edge_fused));
      edge_gradients.push_back(fused_edge_gradient);
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

//      const Eigen::Vector2d p1d = p1.cast<double>();
//      const auto& s = downscale_factor;

      // Watershed.
      const auto frame_blurred = frame.convert<Rgb32f>()
                                      .compute<Gaussian>(1.2f)
                                      .convert<Rgb8>();
      const auto color_threshold = std::sqrt(std::powf(1, 2) * 3);
      const auto regions = color_watershed(frame_blurred, color_threshold);

      // Display the good regions.
      const auto colors = mean_colors(regions, frame);
      auto detection = Image<Rgb8>{frame.sizes()};
      for (const auto& [label, points] : regions)
      {
        // Show big segments only.
        for (const auto& p : points)
          detection(p) = points.size() < 100 ? Black8 : colors.at(label);
      }
      display(detection);

      for (const auto& g : edge_groups)
      {
        for (const auto& e : g.second)
        {
          const auto& edge_refined = edges_refined[e];
          if (edge_refined.size() < 2)
            continue;

          const auto& color = edge_colors[e];
          //draw_polyline(detection, edge_refined, color, p1d, s);
          for (const auto& p: edge_chains[e])
            detection(p) = color;

// #define DEBUG_SHAPE_STATISTICS
#ifdef DEBUG_SHAPE_STATISTICS
          const auto& rect = OrientedBox{.center = centers[e],  //
                                         .axes = axes[e],       //
                                         .lengths = lengths[e]};
          rect.draw(detection, White8, p1d, s);
#endif
        }
      }

      display(detection);

#ifdef SHOW_MEAN_IMAGE_GRADIENTS // I am convinced about their robustness.
      for (const auto& g : edge_groups)
      {
        for (const auto& e : g.second)
        {
          const auto& edge_refined = edges_refined[e];
          if (edge_refined.size() < 2)
            continue;

          const auto& g = mean_gradients[e];
          const Eigen::Vector2f a =
              std::accumulate(edge_refined.begin(), edge_refined.end(),
                              Eigen::Vector2f(0, 0),
                              [](const auto& a, const auto& b) {
                                return a + b.template cast<float>();
                              }) /
              edge_refined.size();
          const Eigen::Vector2f b = a + 20 * g;

          const auto& color = edge_colors[e];
          draw_arrow(a, b, color, 2);
        }
      }
#endif
    };

    tic();
    draw_task();
    // TODO:
    // 1. reject groups that cannot be approximated very well with lines.
    // 2. improve the rejection method by implement the ellipse fitting method
    //    of http://iim.cs.tut.ac.jp/member/kanatani/papers/csdellipse3.pdf

    tic();
    draw_task();
    toc("Draw");
  }

}  // namespace DO::Sara
