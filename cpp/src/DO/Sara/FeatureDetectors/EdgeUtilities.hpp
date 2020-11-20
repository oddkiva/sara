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

#include <DO/Sara/Core/TicToc.hpp>

#include <DO/Sara/Graphics/ImageDraw.hpp>


namespace DO::Sara {

  inline auto
  to_dense_map(const std::map<int, std::vector<Eigen::Vector2i>>& contours,
               const Eigen::Vector2i& image_sizes)
  {
    auto labeled_edges = Image<int>{image_sizes};
    labeled_edges.flat_array().fill(-1);
    for (const auto& [label, points] : contours)
    {
      for (const auto& p : points)
        labeled_edges(p) = label;
    }
    return labeled_edges;
  }

  inline auto
  random_colors(const std::map<int, std::vector<Eigen::Vector2i>>& contours)
  {
    auto colors = std::map<int, Rgb8>{};
    for (const auto& c : contours)
      colors[c.first] = Rgb8(rand() % 255, rand() % 255, rand() % 255);
    return colors;
  }

  template <typename Point>
  auto draw_polyline(ImageView<Rgb8>& image, const std::vector<Point>& edge,
                     const Rgb8& color, const Point& offset = Point::Zero(),
                     float scale = 1)
  {
    auto remap = [&](const auto& p) { return offset + scale * p; };

    for (auto i = 0u; i < edge.size() - 1; ++i)
    {
      const auto& a = remap(edge[i]).template cast<int>();
      const auto& b = remap(edge[i + 1]).template cast<int>();
      draw_line(image, a.x(), a.y(), b.x(), b.y(), color, 1, true);
      fill_circle(image, a.x(), a.y(), 2, color);
      if (i == edge.size() - 2)
        fill_circle(image, b.x(), b.y(), 2, color);
    }
  }

}  // namespace DO::Sara
