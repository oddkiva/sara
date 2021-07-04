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

#include <DO/Sara/ImageProcessing.hpp>


namespace DO::Sara {

  struct Saddle
  {
    Eigen::Vector2i p;
    float score;

    auto operator<(const Saddle& other) const
    {
      return score < other.score;
    }
  };

  inline auto extract_saddle_points(const Image<float>& hessian, float thres)
  {
    auto saddle_points = std::vector<Saddle>{};
    saddle_points.reserve(std::max(hessian.width(), hessian.height()));
    for (auto y = 0; y < hessian.height(); ++y)
      for (auto x = 0; x < hessian.width(); ++x)
        if (hessian(x, y) < thres)
          saddle_points.push_back({{x, y}, std::abs(hessian(x, y))});

    return saddle_points;
  }

  inline auto nms(std::vector<Saddle>& saddle_points,
                  const Eigen::Vector2i& image_sizes, int nms_radius)
  {
    std::sort(saddle_points.begin(), saddle_points.end());

    auto saddle_points_filtered = std::vector<Saddle>{};
    saddle_points_filtered.reserve(saddle_points.size());

    auto saddle_map = Image<std::uint8_t>{image_sizes};
    saddle_map.flat_array().fill(0);

    for (const auto& p : saddle_points)
    {
      if (saddle_map(p.p) == 1)
        continue;

      const auto vmin =
          std::clamp(p.p.y() - nms_radius, 0, saddle_map.height());
      const auto vmax =
          std::clamp(p.p.y() + nms_radius, 0, saddle_map.height());
      const auto umin = std::clamp(p.p.x() - nms_radius, 0, saddle_map.width());
      const auto umax = std::clamp(p.p.x() + nms_radius, 0, saddle_map.width());
      for (auto v = vmin; v < vmax; ++v)
        for (auto u = umin; u < umax; ++u)
          saddle_map(u, v) = 1;

      saddle_points_filtered.push_back(p);
    }
    saddle_points_filtered.swap(saddle_points);
  }

  inline auto detect_saddle_points(const Image<float>& image_blurred,
                                   int nms_radius)
  {
    // Calculate the first derivative.
    const auto gradient = image_blurred.compute<Gradient>();

    // Chessboard corners are saddle points of the image, which are
    // characterized by the property det(H(x, y)) < 0.
    const auto det_hessian =
        image_blurred.compute<Hessian>().compute<Determinant>();

    // Adaptive thresholding.
    const auto thres = det_hessian.flat_array().minCoeff() * 0.05f;
    auto saddle_points = extract_saddle_points(det_hessian, thres);

    // Non-maxima suppression.
    nms(saddle_points, image_blurred.sizes(), nms_radius);

    return saddle_points;
  }

}  // namespace DO::Sara
