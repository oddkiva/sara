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

#include <DO/Sara/ImageProcessing/EdgeDetection.hpp>


namespace DO::Sara {

  //! @brief Group edgels into **unordered** quasi-straight curves.
  auto perform_hysteresis_and_grouping(ImageView<std::uint8_t>& edges,  //
                                       const ImageView<float>& orientations,
                                       float angular_threshold)
      -> std::map<int, std::vector<Point2i>>
  {
    const auto index = [&edges](const Eigen::Vector2i& p) {
      return p.y() * edges.width() + p.x();
    };

    const auto is_strong_edgel = [&edges](const Eigen::Vector2i& p) {
      return edges(p) == 255;
    };

    const auto is_weak_edgel = [&edges](const Eigen::Vector2i& p) {
      return edges(p) == 127;
    };

    const auto orientation_vector = [&orientations](const Vector2i& p) {
      const auto& o = orientations(p);
      return Eigen::Vector2f{cos(o), sin(o)};
    };

    tic();
    const auto sin_threshold = std::sin(angular_threshold);
    const auto angular_distance = [](const auto& a, const auto& b) {
      const auto s = a.homogeneous().cross(b.homogeneous())(2);
      return std::abs(s);
    };

    auto ds = DisjointSets(edges.size());
    auto visited = Image<std::uint8_t>{edges.sizes()};
    visited.flat_array().fill(0);

    // Collect the edgels and make as many sets as pixels.
    auto q = std::queue<Eigen::Vector2i>{};
    for (auto y = 0; y < edges.height(); ++y)
    {
      for (auto x = 0; x < edges.width(); ++x)
      {
        ds.make_set(index({x, y}));
        if (is_strong_edgel({x, y}))
          q.emplace(x, y);
      }
    }

    // Neighborhood defined by 8-connectivity.
    static const auto dir = std::array<Eigen::Vector2i, 8>{
        Eigen::Vector2i{1, 0},    //
        Eigen::Vector2i{1, 1},    //
        Eigen::Vector2i{0, 1},    //
        Eigen::Vector2i{-1, 1},   //
        Eigen::Vector2i{-1, 0},   //
        Eigen::Vector2i{-1, -1},  //
        Eigen::Vector2i{0, -1},   //
        Eigen::Vector2i{1, -1}    //
    };

    // TODO:
    // - just apply the parallel connected component algorithm.
    // - if connected weak edgels don't get contaminated by a strong edgel in
    //   their vicinity, it can be discarded by post-processing.
    //   This is the key idea as for why we can get rid of the queue.
    while (!q.empty())
    {
      const auto& p = q.front();
      visited(p) = 2;  // 2 = visited

      if (!is_strong_edgel(p) && !is_weak_edgel(p))
        throw std::runtime_error{"NOT AN EDGEL!"};

      // Find its corresponding node in the disjoint set.
      const auto node_p = ds.node(index(p));
      const auto up = orientation_vector(p);

      // Add nonvisited weak edges.
      for (const auto& d : dir)
      {
        const Eigen::Vector2i n = p + d;
        // Boundary conditions.
        if (n.x() < 0 || n.x() >= edges.width() ||  //
            n.y() < 0 || n.y() >= edges.height())
          continue;

        // Make sure that the neighbor is an edgel.
        if (!is_strong_edgel(n) && !is_weak_edgel(n))
          continue;

        const auto un = orientation_vector(n);

        // Merge component of p and component of n if angularly consistent.
        if (angular_distance(up, un) < sin_threshold)
        {
          const auto node_n = ds.node(index(n));
          ds.join(node_p, node_n);
        }

        // Enqueue the neighbor n if it is not already enqueued
        if (visited(n) == 0)
        {
          // Enqueue the neighbor.
          q.emplace(n);
          visited(n) = 1;  // 1 = enqueued
          edges(n) = 255;  // Promote to strong edgel!
        }
      }

      q.pop();
    }
    toc("Serial Connected Components");

    tic();
    auto contours = std::map<int, std::vector<Point2i>>{};
    for (auto y = 0; y < edges.height(); ++y)
    {
      for (auto x = 0; x < edges.width(); ++x)
      {
        const auto p = Eigen::Vector2i{x, y};
        const auto index_p = index(p);
        if (is_strong_edgel(p))
        {
          const auto component_id = static_cast<int>(ds.component(index_p));
          if (component_id < 0 ||
              component_id >= edges.width() * edges.height())
            throw std::runtime_error{"Just noooooooo!!!!!!" +
                                     std::to_string(component_id)};
          auto ci = contours.find(component_id);
          if (ci == contours.end())
            contours[component_id] = {p};
          else
            ci->second.push_back(p);
        }
      }
    }
    toc("Serial Contour Collection");

    SARA_DEBUG << "#{contours} = " << contours.size() << std::endl;

    return contours;
  }

}  // namespace DO::Sara
