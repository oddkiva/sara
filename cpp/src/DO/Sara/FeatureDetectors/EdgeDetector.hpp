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

#include <DO/Sara/FeatureDetectors/EdgeDetectionUtilities.hpp>


namespace DO::Sara {

  struct EdgeDetector
  {
    //! @brief intermediate data.
    struct Pipeline
    {
      Image<Vector2f> gradient_cartesian;
      Image<float> gradient_magnitude;
      Image<float> gradient_orientation;

      Image<std::uint8_t> edge_map;
      std::map<int, std::vector<Point2i>> contours;

      std::vector<std::vector<Eigen::Vector2i>> curve_list;
      std::vector<int> curve_ids;
    } pipeline;

    struct Parameters
    {
      //! @brief Canny edge parameters.
      //! @{
      float high_threshold_ratio = 5e-2f;
      float low_threshold_ratio = 2e-2f;
      //! @}

      //! @brief Angle tolerance for connected edgel grouping.
      float angular_threshold = 20. / 180.f * M_PI;
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
      const auto labeled_curves =
          to_map(pipeline.contours, pipeline.edge_map.sizes());

      pipeline.curve_list.clear();
      pipeline.curve_ids.clear();
      for (const auto& [id, curve] : pipeline.contours)
      {
        pipeline.curve_list.emplace_back(curve);
        pipeline.curve_ids.emplace_back(id);
      }
    }
  };

}  // namespace DO::Sara
