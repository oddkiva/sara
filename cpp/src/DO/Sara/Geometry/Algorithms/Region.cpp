// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Geometry/Algorithms/Region.hpp>


using namespace std;


namespace DO { namespace Sara {

  vector<Point2i>
  compute_region_inner_boundary(const ImageView<int>& regions, int region_id)
  {
    // clang-format off
#ifndef CONNECTIVITY_4
    static const auto dirs = std::array{
      Vector2i{ 1,  0 },
      Vector2i{ 1,  1 },
      Vector2i{ 0,  1 },
      Vector2i{-1,  1 },
      Vector2i{-1,  0 },
      Vector2i{-1, -1 },
      Vector2i{ 0, -1 },
      Vector2i{ 1, -1 }
    };
#else
    static const auto dirs = std::array{
      Vector2i{ 1,  0 },
      Vector2i{ 0,  1 },
      Vector2i{-1,  0 },
      Vector2i{ 0, -1 }
    };
#endif
    // clang-format on

    static const auto num_dirs = static_cast<int>(dirs.size());

    // Find the starting point.
    auto start_point = Point2i{ -1, -1 };
    for (int y = 0; y < regions.height(); ++y)
    {
      for (int x = 0; x < regions.width(); ++x)
      {
        if (regions(x, y) == region_id)
        {
          start_point = Point2i(x, y);
          break;
        }
      }
    }
    if (start_point == Point2i{ -1, -1 })
      return {};

    auto boundary = vector<Point2i>{ start_point };

#ifndef CONNECTIVITY_4
    int dir = 7;
#else
    int dir = 0;
#endif
    do {
      const Point2i& current_point{ boundary.back() };

#ifndef CONNECTIVITY_4
      dir = (dir % 2) == 0 ? (dir + 7) % 8 : (dir + 6) % 8;
#else
      dir = (dir + 3) % 4;
#endif
      for (int d = 0; d < num_dirs; ++d)
      {
        Point2i next_point{ current_point + dirs[(dir + d) % num_dirs] };

        if (next_point.minCoeff() < 0 ||
            (regions.sizes() - next_point).minCoeff() <= 0)
          continue;

        if (regions(next_point) == region_id)
        {
          boundary.push_back(next_point);
          dir = (dir + d) % num_dirs;
          break;
        }
      }
    } while (boundary.back() != start_point);
    if (boundary.size() > 1)
      boundary.pop_back();

    return boundary;
  }

  vector<vector<Point2i>>
  compute_region_inner_boundaries(const ImageView<int>& regions)
  {
    const auto num_regions = regions.flat_array().maxCoeff() + 1;
    auto region_boundaries = vector<vector<Point2i>>(num_regions);

    for (auto region_id = 0; region_id < num_regions; ++region_id)
      region_boundaries[region_id] = compute_region_inner_boundary(regions, region_id);

    return region_boundaries;
  }

} /* namespace Sara */
} /* namespace DO */
