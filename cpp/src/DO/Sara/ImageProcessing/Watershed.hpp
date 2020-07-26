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

#include <DO/Sara/DisjointSets/DisjointSets.hpp>

#include <array>
#include <queue>


namespace DO { namespace Sara {

  inline auto color_watershed(                                //
      const ImageView<Rgb8>& image,                           //
      float color_threshold = std::sqrt(std::pow(2, 2) * 3))  //
  {
    const auto squared_color_threshold = std::pow(color_threshold, 2);
    const auto index = [&image](const Eigen::Vector2i& p) {
      return p.y() * image.width() + p.x();
    };

    auto ds = DisjointSets(image.size());

    // Make as many sets as pixels.
    for (auto y = 0; y < image.height(); ++y)
      for (auto x = 0; x < image.width(); ++x)
        ds.make_set(index({x, y}));

    for (auto y = 0; y < image.height(); ++y)
    {
      for (auto x = 0; x < image.width(); ++x)
      {
        // Find its corresponding node in the disjoint set.
        const auto p = Eigen::Vector2i{x, y};
        const auto node_p = ds.node(index(p));

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
              const auto node_n = ds.node(index(n));
              ds.join(node_p, node_n);
            }
          }
        }
      }
    }

    auto regions = std::map<int, std::vector<Point2i>>{};
    for (auto y = 0; y < image.height(); ++y)
    {
      for (auto x = 0; x < image.width(); ++x)
      {
        const auto p = Eigen::Vector2i{x, y};
        const auto index_p = index(p);
        regions[ds.component(index_p)].push_back(p);
      }
    }

    return regions;
  }

}}  // namespace DO::Sara
