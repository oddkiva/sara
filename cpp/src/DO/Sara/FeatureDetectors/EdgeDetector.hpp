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

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Sara {

  struct DO_SARA_EXPORT EdgeDetector
  {
    //! @brief intermediate data.
    struct Pipeline
    {
      Image<Vector2f> gradient_cartesian;
      Image<float> gradient_magnitude;
      Image<float> gradient_orientation;

      Image<std::uint8_t> edge_map;
      std::map<int, std::vector<Point2i>> edges;

      std::vector<std::vector<Eigen::Vector2i>> edges_as_list;
      std::vector<std::vector<Eigen::Vector2d>> edges_simplified;

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

      //! @brief Edge simplification parameters.
      bool simplify_edges = true;
      double eps = 1.;
      double collapse_threshold = 2e-2;
      bool collapse_adaptive = true;
    } parameters;

    EdgeDetector() = default;

    EdgeDetector(const EdgeDetector::Parameters& params)
      : parameters{params}
    {
    }

    auto operator()(const Image<float>& image) -> void;
  };

}  // namespace DO::Sara
