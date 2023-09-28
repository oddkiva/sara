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
#include <DO/Sara/DisjointSets/DisjointSetsV2.hpp>

#include <array>
#include <queue>


namespace DO::Sara { namespace v2 {

  inline auto color_watershed(                                //
      const ImageView<Rgb8>& image,                           //
      float color_threshold = std::sqrt(std::pow(2, 2) * 3))  //
  {
    const auto squared_color_threshold = std::pow(color_threshold, 2);
    const auto index = [&image](const Eigen::Vector2i& p) {
      return p.y() * image.width() + p.x();
    };

    tic();
    auto ds = v2::DisjointSets(static_cast<std::uint32_t>(image.size()));

    const auto w = image.width();
    const auto h = image.height();
    const auto wh = w * h;
#pragma omp parallel for
    for (auto xy = 0; xy < wh; ++xy)
    {
      const auto y = xy / w;
      const auto x = xy - y * w;

      // Find its corresponding node in the disjoint set.
      const auto p = Eigen::Vector2i{x, y};
      const auto node_p = index(p);

      const Vector3f& color_p = image(p).cast<float>();

      for (auto v = 0; v <= 1; ++v)
      {
        for (auto u = 0; u <= 1; ++u)
        {
          if (u == 0 && v == 0)
            continue;

          const Eigen::Vector2i& n = p + Eigen::Vector2i{u, v};
          // Boundary conditions.
          if (n.x() >= image.width() || n.y() >= image.height())
            continue;

          const Vector3f& color_n = image(n).cast<float>();

          const auto dist = (color_p - color_n).squaredNorm();

          // Merge component of p and component of n if their colors are
          // close.
          if (dist < squared_color_threshold)
          {
            const auto node_n = index(n);
#pragma omp critical
            ds.join(node_p, node_n);
          }
        }
      }
    }
    toc("Connected components V2");

    tic();
    auto regions = std::vector<std::vector<Point2i>>(image.size());
#pragma omp parallel for
    for (auto xy = 0; xy < wh; ++xy)
    {
      const auto y = xy / w;
      const auto x = xy - y * w;
      const auto p = Eigen::Vector2i{x, y};
      const auto index_p = index(p);
#pragma omp critical
      regions[ds.parent(index_p)].push_back(p);
    }
    toc("Region Collection V2");

    return regions;
  }

}}  // namespace DO::Sara::v2
