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

#include <DO/Sara/Core/PhysicalQuantities.hpp>
#include <DO/Sara/FeatureDetectors/EdgePostProcessing.hpp>


namespace DO::Sara {

  struct LineSegmentDetector
  {
    //! @brief intermediate data.
    struct Pipeline
    {
      Image<Eigen::Vector2f> gradient_cartesian;
      Image<float> gradient_magnitude;
      Image<float> gradient_orientation;
      Image<std::uint8_t> edge_map;
      std::map<int, std::vector<Point2i>> contours;
      std::vector<std::vector<Point2i>> curve_list;
      std::vector<int> curve_ids;
      std::vector<std::tuple<bool, LineSegment>> line_segments_candidates;
      std::vector<std::tuple<int, LineSegment>> line_segments;
    } pipeline;

    struct Parameters
    {
      //! @brief Canny edge parameters.
      //! @{
      float high_threshold_ratio = 5e-2f;
      float low_threshold_ratio = 2e-2f;
      //! @}

      //! @brief Angle tolerance for connected edgel grouping.
      float angular_threshold = static_cast<float>(20._deg);

      //! @brief RANSAC-based parameters for line segment fitting.
      //! @{
      int num_iteration_min = 5;
      int num_iteration_max = 20;
      float iteration_percentage = 0.20f;
      bool polish_line_segments = true;
      //! @}
    } parameters;

    auto operator()(const Image<float>& image)
    {
      pipeline.gradient_cartesian = gradient(image);
      pipeline.gradient_magnitude = pipeline.gradient_cartesian.cwise_transform(
          [](const auto& v) { return v.norm(); });
      pipeline.gradient_orientation =
          pipeline.gradient_cartesian.cwise_transform(
              [](const auto& v) { return std::atan2(v.y(), v.x()); });

      const auto& grad_mag_max =
          pipeline.gradient_magnitude.flat_array().maxCoeff();
      const auto& high_thres = grad_mag_max * parameters.high_threshold_ratio;
      const auto& low_thres = grad_mag_max * parameters.low_threshold_ratio;

      pipeline.edge_map = suppress_non_maximum_edgels(  //
          pipeline.gradient_magnitude,                  //
          pipeline.gradient_orientation,                //
          high_thres, low_thres);
      hysteresis(pipeline.edge_map);

      // Extract quasi-straight curve.
      pipeline.contours = connected_components(  //
          pipeline.edge_map,                     //
          pipeline.gradient_orientation,         //
          parameters.angular_threshold);

      pipeline.curve_list.clear();
      pipeline.curve_ids.clear();
      for (const auto& [id, curve] : pipeline.contours)
      {
        pipeline.curve_list.emplace_back(curve);
        pipeline.curve_ids.emplace_back(id);
      }

      // Fit a line to each curve.
      pipeline.line_segments_candidates =
          std::vector<std::tuple<bool, LineSegment>>(
              pipeline.curve_list.size(),
              {false, {}}
          );

#pragma omp parallel for
      for (auto i = 0; i < static_cast<int>(pipeline.curve_list.size()); ++i)
      {
        const auto& curve = pipeline.curve_list[i];
        if (curve.size() < 2)
          continue;

        auto num_iterations =
            static_cast<int>(curve.size() * parameters.iteration_percentage) +
            1;
        num_iterations = std::clamp(       //
            num_iterations,                //
            parameters.num_iteration_min,  //
            parameters.num_iteration_max);
        pipeline.line_segments_candidates[i] = fit_line_segment_robustly(  //
            curve,                                                         //
            num_iterations,                                                //
            parameters.polish_line_segments,                               //
            /* line_fit_thresh */ 1.f);                                    //
      }

      // Filter the line segment candidates.
      pipeline.line_segments.reserve(pipeline.curve_ids.size());
      for (auto i = 0u; i < pipeline.curve_ids.size(); ++i)
      {
        const auto& curve_id = pipeline.curve_ids[i];
        const auto& [is_line_segment, line_segment] =
            pipeline.line_segments_candidates[i];

        if (!is_line_segment)
          continue;
        pipeline.line_segments.push_back({curve_id, line_segment});
      }
    }
  };

}  // namespace DO::Sara
