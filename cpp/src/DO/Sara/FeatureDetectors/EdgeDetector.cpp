// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/FeatureDetectors/EdgeDetector.hpp>
#include <DO/Sara/FeatureDetectors/EdgePostProcessing.hpp>
#include <DO/Sara/Geometry/Algorithms/RamerDouglasPeucker.hpp>


namespace DO::Sara {

  auto EdgeDetector::operator()(const Image<float>& image) -> void
  {
    tic();
    pipeline.gradient_cartesian = gradient(image);
    toc("Gradient");

    tic();
    pipeline.gradient_magnitude = pipeline.gradient_cartesian.cwise_transform(
        [](const auto& v) { return v.norm(); });
    pipeline.gradient_orientation = pipeline.gradient_cartesian.cwise_transform(
        [](const auto& v) { return std::atan2(v.y(), v.x()); });
    toc("Polar Coordinates");

    tic();
    const auto& grad_mag = pipeline.gradient_magnitude;
    const auto& grad_mag_max = grad_mag.flat_array().maxCoeff();
    const auto& high_thres = grad_mag_max * parameters.high_threshold_ratio;
    const auto& low_thres = grad_mag_max * parameters.low_threshold_ratio;
    pipeline.edge_map = suppress_non_maximum_edgels(  //
        pipeline.gradient_magnitude,                  //
        pipeline.gradient_orientation,                //
        high_thres, low_thres);
    toc("Thresholding");

    tic();
    pipeline.edges = perform_hysteresis_and_grouping(  //
        pipeline.edge_map,                             //
        pipeline.gradient_orientation,                 //
        parameters.angular_threshold);
    toc("Hysteresis & Edgel Grouping");

    tic();
    const auto& edges = pipeline.edges;
    auto& edges_as_list = pipeline.edges_as_list;
    edges_as_list.resize(edges.size());
    std::transform(edges.begin(), edges.end(), edges_as_list.begin(),
                   [](const auto& e) { return e.second; });
    toc("To vector");

    if (parameters.simplify_edges)
    {
      tic();
      auto& edges_simplified = pipeline.edges_simplified;
      edges_simplified.resize(edges_as_list.size());
#pragma omp parallel for
      for (auto i = 0u; i < edges_as_list.size(); ++i)
      {
        const auto& edge = reorder_and_extract_longest_curve(edges_as_list[i]);

        auto edges_converted = std::vector<Eigen::Vector2d>(edge.size());
        std::transform(edge.begin(), edge.end(), edges_converted.begin(),
                       [](const auto& p) { return p.template cast<double>(); });

        edges_simplified[i] = ramer_douglas_peucker(edges_converted, parameters.eps);
      }
      toc("Longest Curve Extraction & Simplification");

//       tic();
// #pragma omp parallel for
//       for (auto i = 0u; i < edges_simplified.size(); ++i)
//         if (edges_simplified[i].size() > 2)
//           edges_simplified[i] = collapse(edges_simplified[i], grad_mag,
//                                          parameters.collapse_threshold,
//                                          parameters.collapse_adaptive);
//       toc("Vertex Collapse");
// 
//       tic();
//       auto& edges_refined = edges_simplified;
// #pragma omp parallel for
//       for (auto i = 0u; i < edges_refined.size(); ++i)
//         for (auto& p : edges_refined[i])
//           p = refine(grad_mag, p.cast<int>()).cast<double>();
//       toc("Refine Edge Localisation");
     }
  }

}  // namespace DO::Sara
