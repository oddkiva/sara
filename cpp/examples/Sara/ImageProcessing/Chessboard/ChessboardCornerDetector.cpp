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

#include "SaddlePointDetection.hpp"


namespace DO::Sara {

  auto weight_mask(const std::vector<int>& radius)
  {
    std::unordered_map<int, Image<float>> masks;
    static constexpr auto ring_radius = 0.3f;
    static constexpr auto dist_min = 1 - ring_radius;
    static constexpr auto dist_max = 1 + ring_radius;
    static constexpr auto weight_typical = 0.6f;

    for (const auto& r : radius)
    {
      auto& mask = masks[r];
      mask.resize(2 * r + 1, 2 * r + 1);
      for (auto v = 0; v < mask.height(); ++v)
      {
        for (auto u = 0; u < mask.width(); ++u)
        {
          // Calculate the normalized distance.
          const auto dist =
              (Eigen::Vector2f(u, v) - r * Eigen::Vector2f::Ones()).norm() / r;
          const auto clamped_dist = std::clamp(dist, dist_min, dist_max);
          // Linear function.
          mask(u, v) = (dist_max - clamped_dist) / weight_typical;
        }
      }
    }
    return masks;
  }

}  // namespace DO::Sara
