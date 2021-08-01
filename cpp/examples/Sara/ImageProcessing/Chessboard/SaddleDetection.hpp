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

#include <DO/Sara/Graphics.hpp>
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

  inline auto extract_saddle_points(const Image<float>& det_of_hessian,
                                    const Image<Eigen::Matrix2f>& hessian,
                                    float thres)
  {
    auto saddle_points = std::vector<Saddle>{};
    saddle_points.reserve(std::max(hessian.width(), hessian.height()));

    for (auto y = 0; y < hessian.height(); ++y)
    {
      for (auto x = 0; x < hessian.width(); ++x)
      {
        if (det_of_hessian(x, y) > thres)
          continue;

        auto qr_factorizer = hessian(x, y).householderQr();
        const Eigen::Matrix2f R = qr_factorizer.matrixQR().triangularView<Eigen::Upper>();

        if (std::abs((R(0, 0) - R(1, 1)) / R(0, 0)) > 0.1f)
          continue;

        saddle_points.push_back({{x, y}, std::abs(det_of_hessian(x, y))});
      }
    }

    return saddle_points;
  }

  inline auto nms(std::vector<Saddle>& saddle_points,
                  const Eigen::Vector2i& image_sizes,
                  int nms_radius)
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

  inline auto detect_saddle_points(const Image<float>& image,
                                   int nms_radius,
                                   float adaptive_thres = 0.05f)
  {
    // Calculate the first derivative.
    const auto hessian = image.compute<Hessian>();

    // Chessboard corners are saddle points of the image, which are
    // characterized by the property det(H(x, y)) < 0.
    const auto det_of_hessian = hessian.compute<Determinant>();

    // Adaptive thresholding.
    const auto thres = det_of_hessian.flat_array().minCoeff() * adaptive_thres;
    auto saddle_points = extract_saddle_points(det_of_hessian, hessian, thres);

    // Non-maxima suppression.
    nms(saddle_points, image.sizes(), nms_radius);

    return saddle_points;
  }

}  // namespace DO::Sara
