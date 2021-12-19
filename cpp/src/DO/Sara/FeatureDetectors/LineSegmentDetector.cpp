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

#include <DO/Sara/FeatureDetectors/LineSegmentDetector.hpp>


namespace DO::Sara {

  auto LineSegmentDetector::operator()(const Image<float>& image) -> void
  {
    if (pipeline.gradient_magnitude.sizes() != image.sizes())
      pipeline.gradient_magnitude.resize(image.sizes());
    if (pipeline.gradient_orientation.sizes() != image.sizes())
      pipeline.gradient_orientation.resize(image.sizes());
    gradient_in_polar_coordinates(image, pipeline.gradient_magnitude,
                                  pipeline.gradient_orientation);

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
        std::vector<std::tuple<bool, LineSegment>>(pipeline.curve_list.size(),
                                                   {false, {}});

#pragma omp parallel for
    for (auto i = 0; i < static_cast<int>(pipeline.curve_list.size()); ++i)
    {
      const auto& curve = pipeline.curve_list[i];
      if (curve.size() < 2)
        continue;

      auto num_iterations =
          static_cast<int>(curve.size() * parameters.iteration_percentage) + 1;
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

}  // namespace DO::Sara
