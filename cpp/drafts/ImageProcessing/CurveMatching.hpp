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

#pragma once

#include <Eigen/Core>
#include <DO/Sara/ImageProcessing/EdgeShapeStatistics.hpp>
#include <vector>


namespace DO::Sara {

  //! @brief Embryonary curve matcher class.
  /*!
   *  auto curve_matcher = CurveMatcher{};
   *  curve_matcher.reset_curve_map(frame.width(), frame.height());
   *
   *  // Perform curve matching.
   *  curve_matcher.update_curve_features(edges);
   *
   *  for (const auto& e : edges_refined)
   *    if (e.size() >= 2)
   *      draw_polyline(detection, e, Blue8, p1d, s);
   *  for (auto i = 0u; i < curve_matcher.curves_prev.size(); ++i)
   *  {
   *    const auto& e = curve_matcher.curves_prev[i];
   *    draw_polyline(detection, e, Magenta8, p1d, s);
   *  }
   */
  struct CurveMatcher
  {
    std::vector<std::vector<Eigen::Vector2d>> curves_prev;
    std::vector<std::vector<Eigen::Vector2d>> curves_curr;

    CurveStatistics stats_prev;
    CurveStatistics stats_curr;

    Image<int> curve_map_prev;
    Image<int> curve_map_curr;

    auto reset_curve_map(int w, int h) -> void
    {
      curve_map_prev.resize({w, h});
      curve_map_curr.resize({w, h});

      curve_map_prev.flat_array().fill(-1);
      curve_map_curr.flat_array().fill(-1);
    }

    auto recalculate_curve_map(
        const std::vector<std::vector<Eigen::Vector2i>>& curve_points) -> void
    {
      curve_map_curr.flat_array().fill(-1);
#pragma omp parallel for
      for (auto i = 0; i < static_cast<int>(curve_points.size()); ++i)
      {
        const auto& points = curve_points[i];
        for (const auto& p : points)
          curve_map_curr(p) = i;
      }
    }

    auto update_curve_features(
        const std::vector<std::vector<Eigen::Vector2d>>& curves_as_polylines)
    {
      curves_curr.swap(curves_prev);
      curves_curr = curves_as_polylines;

      stats_curr.swap(stats_prev);
      stats_curr = CurveStatistics(curves_curr);

      curve_map_curr.swap(curve_map_prev);
      // recalculate_curve_map(curve_points);
    }
  };

}  // namespace DO::Sara
