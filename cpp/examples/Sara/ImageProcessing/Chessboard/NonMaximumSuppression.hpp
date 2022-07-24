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

    for (const auto& f : features)
    {
      const Eigen::Vector2i p = f.position().template cast<int>();
      if (feature_map(p) == 1)
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

}  // namespace DO::Sara
