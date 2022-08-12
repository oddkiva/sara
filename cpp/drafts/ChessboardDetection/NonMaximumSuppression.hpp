// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Image.hpp>

#include <vector>


namespace DO::Sara {

  template <typename Feature>
  auto nms(std::vector<Feature>& features, const Eigen::Vector2i& image_sizes,
           int nms_radius) -> void
  {
    std::sort(features.rbegin(), features.rend());

    auto features_filtered = std::vector<Feature>{};
    features_filtered.reserve(features.size());

    auto feature_map = Image<std::uint8_t>{image_sizes};
    feature_map.flat_array().fill(0);

    const auto w = feature_map.width();
    const auto h = feature_map.height();

    for (const auto& f : features)
    {
      const Eigen::Vector2i p = f.position().template cast<int>();
      const auto in_image_domain = 0 <= p.x() && p.x() < w &&  //
                                   0 <= p.y() && p.y() < h;
      if (!in_image_domain || feature_map(p) == 1)
        continue;

      const auto vmin = std::clamp(p.y() - nms_radius, 0, feature_map.height());
      const auto vmax = std::clamp(p.y() + nms_radius, 0, feature_map.height());
      const auto umin = std::clamp(p.x() - nms_radius, 0, feature_map.width());
      const auto umax = std::clamp(p.x() + nms_radius, 0, feature_map.width());
      for (auto v = vmin; v < vmax; ++v)
        for (auto u = umin; u < umax; ++u)
          feature_map(u, v) = 1;

      features_filtered.push_back(f);
    }
    features_filtered.swap(features);
  }

  //! @brief N.B.: this will update the scale of the features!
  template <typename Feature>
  auto scale_aware_nms(std::vector<Feature>& features,
                       const Eigen::Vector2i& image_sizes,
                       const float scale_factor) -> void
  {
    std::sort(features.rbegin(), features.rend());

    auto features_filtered = std::vector<Feature>{};
    features_filtered.reserve(features.size());

    auto feature_map = Image<std::int32_t>{image_sizes};
    feature_map.flat_array().fill(-1);

    const auto w = feature_map.width();
    const auto h = feature_map.height();

    const auto num_features = static_cast<std::int32_t>(features.size());
    for (auto fid = 0; fid < num_features; ++fid)
    {
      const auto& f = features[fid];
      const Eigen::Vector2i p = f.position().template cast<int>();
      const auto in_image_domain = 0 <= p.x() && p.x() < w &&  //
                                   0 <= p.y() && p.y() < h;
      if (!in_image_domain)
        continue;

      const auto best_feature_id = feature_map(p);

      if (best_feature_id == -1)
      {
        // This is the best feature in the vicinity, let's add it.
        const auto nms_radius = static_cast<int>(  //
            std::round(static_cast<float>(M_SQRT2) * f.scale * scale_factor));

        const auto vmin = std::clamp(p.y() - nms_radius, 0, h);
        const auto vmax = std::clamp(p.y() + nms_radius, 0, h);
        const auto umin = std::clamp(p.x() - nms_radius, 0, w);
        const auto umax = std::clamp(p.x() + nms_radius, 0, w);

        for (auto v = vmin; v < vmax; ++v)
          for (auto u = umin; u < umax; ++u)
            feature_map(u, v) = 1;

        features_filtered.push_back(f);
      }
      else
      {
        // Update the scale of the local best corner, it means it is redetected
        // at a coarser scale.
        auto& best_feature = features[best_feature_id];
        if (f.scale < best_feature.scale)
          continue;
        best_feature.scale = f.scale;

        const auto nms_radius = static_cast<int>(  //
            std::round(static_cast<float>(M_SQRT2) * f.scale * scale_factor));

        const auto vmin = std::clamp(p.y() - nms_radius, 0, h);
        const auto vmax = std::clamp(p.y() + nms_radius, 0, h);
        const auto umin = std::clamp(p.x() - nms_radius, 0, w);
        const auto umax = std::clamp(p.x() + nms_radius, 0, w);

        for (auto v = vmin; v < vmax; ++v)
          for (auto u = umin; u < umax; ++u)
            feature_map(u, v) = best_feature_id;
      }
    }
    features_filtered.swap(features);
  }

}  // namespace DO::Sara
