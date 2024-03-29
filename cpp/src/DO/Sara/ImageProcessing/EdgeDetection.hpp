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

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/TicToc.hpp>

#include <DO/Sara/DisjointSets/DisjointSets.hpp>
#include <DO/Sara/DisjointSets/DisjointSetsV2.hpp>

#include <DO/Sara/ImageProcessing/Differential.hpp>
#include <DO/Sara/ImageProcessing/Interpolation.hpp>
#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>
#include <DO/Sara/ImageProcessing/SecondMomentMatrix.hpp>

#include <DO/Sara/Geometry/Objects/LineSegment.hpp>

#include <array>
#include <cstdint>
#include <queue>


namespace DO::Sara {

  // ======================================================================== //
  // Edge Detection Encoded as a Dense Feature Map.
  // ==========================================================================

  //! @brief Building blocks for Canny's edge detector.
  //! @{
  inline auto suppress_non_maximum_edgels(const ImageView<float>& grad_mag,
                                          const ImageView<float>& grad_ori,
                                          const float high_thres,
                                          const float low_thres)
  {
    auto edges = Image<uint8_t>{grad_mag.sizes()};
    edges.flat_array().fill(0);

    const auto w = grad_mag.width();
    const auto h = grad_mag.height();
    const auto wh = w * h;

#define FAST_COS_AND_SIN
#ifdef FAST_COS_AND_SIN
    static constexpr auto fast_cos = [](float x) {
      float u = 1.2467379e-32f;
      u = u * x + -9.6966799e-4f;
      u = u * x + -1.8279663e-31f;
      u = u * x + 3.922768e-2f;
      u = u * x + 7.4160361e-31f;
      u = u * x + -4.9534958e-1f;
      u = u * x + -7.1721109e-31f;
      return u * x + 9.986066e-1f;
    };

    static constexpr auto fast_sin = [](const float x) {
      float u = -1.4507699e-4f;
      u = u * x + -9.7064129e-41f;
      u = u * x + 7.9580618e-3f;
      u = u * x + 1.118603e-39f;
      u = u * x + -1.6566699e-1f;
      u = u * x + -1.7063928e-39f;
      u = u * x + 9.9927587e-1f;
      return u * x + 7.7202328e-42f;
    };
#endif

#pragma omp parallel for
    for (auto xy = 0; xy < wh; ++xy)
    {
      const auto y = xy / w;
      const auto x = xy - y * w;
      if (x == 0 || x == w - 1 || y == 0 || y == h - 1)
        continue;

      const auto& grad_curr = grad_mag(x, y);
      if (grad_curr < low_thres)
        continue;

      const Vector2d p = Vector2i(x, y).cast<double>();
#ifdef FAST_COS_AND_SIN
      static constexpr auto pi = static_cast<float>(M_PI);
      static constexpr auto two_pi = static_cast<float>(2 * M_PI);
      auto theta = grad_ori(x, y);
      if (theta >= pi)
        theta -= two_pi;
      const auto c = fast_cos(theta);
      const auto s = fast_sin(theta);
      const Vector2d d = Vector2f{c, s}.cast<double>().normalized();
#else
      const auto theta = grad_ori(x, y);
      const Vector2d d = Vector2f{cos(theta), sin(theta)}.cast<double>();
#endif
      const Vector2d p0 = p - d;
      const Vector2d p2 = p + d;
      const auto grad_prev = interpolate(grad_mag, p0);
      const auto grad_next = interpolate(grad_mag, p2);

      const auto is_max = grad_curr > grad_prev &&  //
                          grad_curr > grad_next;
      if (!is_max)
        continue;

      edges(x, y) = grad_curr > high_thres ? 255 : 127;
    }
    return edges;
  }

  inline auto hysteresis(ImageView<std::uint8_t>& edges)
  {
    auto visited = Image<std::uint8_t>{edges.sizes()};
    visited.flat_array().fill(0);

    std::queue<Eigen::Vector2i> queue;
    for (auto y = 0; y < edges.height(); ++y)
    {
      for (auto x = 0; x < edges.width(); ++x)
      {
        if (edges(x, y) == 255)
        {
          queue.emplace(x, y);
          visited(x, y) = 1;
        }
      }
    }

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
    while (!queue.empty())
    {
      const auto& p = queue.front();

      // Promote a weak edge to a strong edge.
      if (edges(p) != 255)
        edges(p) = 255;

      // Add nonvisited weak edges.
      for (const auto& d : dir)
      {
        const Eigen::Vector2i n = p + d;
        // Boundary conditions.
        if (n.x() < 0 || n.x() >= edges.width() ||  //
            n.y() < 0 || n.y() >= edges.height())
          continue;

        if (edges(n) == 127 && !visited(n))
        {
          visited(n) = 1;
          queue.emplace(n);
        }
      }

      queue.pop();
    }
  }
  //! @}


  // ======================================================================== //
  // Edge Detection As Feature Detection Where a Curve Is The Feature.
  // ==========================================================================

  //! @brief Group edgels into **unordered** point sets.
  inline auto connected_components(const ImageView<std::uint8_t>& edges)
  {
    const auto index = [&edges](const Eigen::Vector2i& p) {
      return p.y() * edges.width() + p.x();
    };

    const auto is_edgel = [&edges](const Eigen::Vector2i& p) {
      return edges(p) == 255 || edges(p) == 127;
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
        if (is_edgel({x, y}))
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

    while (!q.empty())
    {
      const auto& p = q.front();
      visited(p) = 2;  // 2 = visited

      if (!is_edgel(p))
        throw std::runtime_error{"NOT AN EDGEL!"};

      // Find its corresponding node in the disjoint set.
      const auto node_p = ds.node(index(p));

      // Add nonvisited weak edges.
      for (const auto& d : dir)
      {
        const Eigen::Vector2i n = p + d;
        // Boundary conditions.
        if (n.x() < 0 || n.x() >= edges.width() ||  //
            n.y() < 0 || n.y() >= edges.height())
          continue;

        // Make sure that the neighbor is an edgel.
        if (!is_edgel(n))
          continue;

        // Merge component of p and component of n.
        const auto node_n = ds.node(index(n));
        ds.join(node_p, node_n);

        // Enqueue the neighbor n if it is not already enqueued
        if (visited(n) == 0)
        {
          // Enqueue the neighbor.
          q.emplace(n);
          visited(n) = 1;  // 1 = enqueued
        }
      }

      q.pop();
    }

    auto contours = std::map<int, std::vector<Point2i>>{};
    for (auto y = 0; y < edges.height(); ++y)
    {
      for (auto x = 0; x < edges.width(); ++x)
      {
        const auto p = Eigen::Vector2i{x, y};
        const auto index_p = index(p);
        if (is_edgel(p))
          contours[static_cast<int>(ds.component(index_p))].push_back(p);
      }
    }

    return contours;
  }

  //! @brief Group edgels into **unordered** quasi-straight curves.
  inline auto connected_components(const ImageView<std::uint8_t>& edges,
                                   const ImageView<float>& orientation,
                                   const float angular_threshold)
  {
    const auto index = [&edges](const Eigen::Vector2i& p) {
      return p.y() * edges.width() + p.x();
    };

    const auto is_edgel = [&edges](const Eigen::Vector2i& p) {
      return edges(p) == 255;
    };

    const auto orientation_vector = [&orientation](const Vector2i& p) {
      const auto& o = orientation(p);
      return Eigen::Vector2f{cos(o), sin(o)};
    };

    const auto angular_distance = [](const auto& a, const auto& b) {
      const auto c = a.dot(b);
      const auto s = a.homogeneous().cross(b.homogeneous())(2);
      const auto dist = std::abs(std::atan2(s, c));
      return dist;
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
        if (is_edgel({x, y}))
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

    while (!q.empty())
    {
      const auto& p = q.front();
      visited(p) = 2;  // 2 = visited

      if (!is_edgel(p))
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
        if (!is_edgel(n))
          continue;

        const auto un = orientation_vector(n);

        // Merge component of p and component of n if angularly consistent.
        if (angular_distance(up, un) < angular_threshold)
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
        }
      }

      q.pop();
    }

    auto contours = std::map<int, std::vector<Point2i>>{};
    for (auto y = 0; y < edges.height(); ++y)
    {
      for (auto x = 0; x < edges.width(); ++x)
      {
        const auto p = Eigen::Vector2i{x, y};
        const auto index_p = index(p);
        if (is_edgel(p))
          contours[static_cast<int>(ds.component(index_p))].push_back(p);
      }
    }

    return contours;
  }

  //! @brief Group edgels into **unordered** quasi-straight curves.
  auto perform_hysteresis_and_grouping(ImageView<std::uint8_t>& edges,  //
                                       const ImageView<float>& orientations,
                                       const float angular_threshold)
      -> std::map<int, std::vector<Point2i>>;

}  // namespace DO::Sara
