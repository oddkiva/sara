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

#include <omp.h>

#include "JunctionDetection.hpp"


namespace DO::Sara {

  auto junction_map(const ImageView<float>& image,
                    const ImageView<Eigen::Vector2f>& gradients, const int r)
      -> Image<float>
  {
    auto junction_map = Image<float>{image.sizes()};
    junction_map.flat_array().fill(0);

    const auto w = image.width();
    const auto h = image.height();

#pragma omp parallel for collapse(2)
    for (auto y = 0; y < image.height(); ++y)
    {
      for (auto x = 0; x < image.width(); ++x)
      {
        const auto in_valid_domain = r <= x && x < w - r &&  //
                                     r <= y && y < h - r;
        if (!in_valid_domain)
          continue;

        auto score = float{};

        const auto p = Eigen::Vector2i{x, y};

        for (auto v = -r; v <= r; ++v)
        {
          for (auto u = -r; u <= r; ++u)
          {
            const auto q = Eigen::Vector2i{x + u, y + v};
            score += square((q - p).cast<float>().dot(gradients(q)));
          }
        }

        junction_map(x, y) = score;
      }
    }

    return junction_map;
  }

  auto extract_junctions(const ImageView<float>& junction_map, const int radius)
      -> std::vector<Junction<int>>
  {
    auto junctions = std::vector<Junction<int>>{};

    static const auto is_local_min = LocalExtremum<std::less_equal, float>{};
    const auto w = junction_map.width();
    const auto h = junction_map.height();

    for (int y = radius; y < h - radius; ++y)
      for (int x = radius; x < w - radius; ++x)
        if (is_local_min(x, y, junction_map))
          junctions.push_back({Eigen::Vector2i(x, y), junction_map(x, y)});

    return junctions;
  }

}  // namespace DO::Sara
