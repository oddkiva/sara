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

    auto operator()(const Image<float>& image) -> void;
  };

}  // namespace DO::Sara
