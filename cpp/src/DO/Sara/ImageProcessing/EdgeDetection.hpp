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

#include <DO/Sara/DisjointSets/DisjointSets.hpp>

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
                                          float high_thres, float low_thres)
  {
    auto edges = Image<uint8_t>{grad_mag.sizes()};
    edges.flat_array().fill(0);
#pragma omp parallel for
    for (auto y = 1; y < grad_mag.height() - 1; ++y)
    {
      for (auto x = 1; x < grad_mag.width() - 1; ++x)
      {
        const auto& grad_curr = grad_mag(x, y);
        if (grad_curr < low_thres)
          continue;

        const auto& theta = grad_ori(x, y);
        const Vector2d p = Vector2i(x, y).cast<double>();
        const Vector2d d = Vector2f{cos(theta), sin(theta)}.cast<double>();
        const Vector2d p0 = p - d;
        const Vector2d p2 = p + d;
        const auto grad_prev = interpolate(grad_mag, p0);
        const auto grad_next = interpolate(grad_mag, p2);

        const auto is_max = grad_curr > grad_prev &&  //
                            grad_curr > grad_next;
        if (!is_max)
          continue;

        edges(x, y) = grad_curr > high_thres ? 255 : 128;
      }
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

    const auto dir = std::array<Eigen::Vector2i, 8>{
        Eigen::Vector2i{1, 0},  Eigen::Vector2i{1, 1},  Eigen::Vector2i{0, 1},
        Eigen::Vector2i{-1, 1}, Eigen::Vector2i{-1, 0}, Eigen::Vector2i{-1, -1},
        Eigen::Vector2i{0, -1}, Eigen::Vector2i{1, -1}};
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

        if (edges(n) == 128 && !visited(n))
        {
          visited(n) = 1;
          queue.emplace(n);
        }
      }

      queue.pop();
    }
  }
  //! @}

  //! @brief Calculate the edge map using Canny operator.
  inline auto canny(const ImageView<float>& frame_gray32f,
                    float high_threshold_ratio = 2e-2f,
                    float low_threshold_ratio = 1e-2f)
  {
    if (!(low_threshold_ratio < high_threshold_ratio &&
          high_threshold_ratio < 1))
      throw std::runtime_error{"Invalid threshold ratios!"};

    const auto& grad = gradient(frame_gray32f);
    const auto& grad_mag = grad.cwise_transform(  //
        [](const auto& v) { return v.norm(); });
    const auto& grad_ori = grad.cwise_transform(
        [](const auto& v) { return std::atan2(v.y(), v.x()); });

    const auto& grad_mag_max = grad_mag.flat_array().maxCoeff();
    const auto& high_thres = grad_mag_max * high_threshold_ratio;
    const auto& low_thres = grad_mag_max * low_threshold_ratio;

    auto edges = suppress_non_maximum_edgels(grad_mag, grad_ori,  //
                                             high_thres, low_thres);
    hysteresis(edges);

    return edges;
  }

  //! @brief Calculate Harris' cornerness function.
  inline auto harris_cornerness_function(const ImageView<float>& I,
                                         float kappa = 0.04f, float sigma = 3.f)
  {
    return I
        .compute<Gradient>()            //
        .compute<SecondMomentMatrix>()  //
        .compute<Gaussian>(sigma)       //
        .cwise_transform([kappa](const auto& m) {
          return m.determinant() - kappa * std::pow(m.trace(), 2);
        });
  }


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
      return edges(p) == 255 || edges(p) == 128;
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
    const auto dir = std::array<Eigen::Vector2i, 8>{
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
                                   float angular_threshold)
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
    const auto dir = std::array<Eigen::Vector2i, 8>{
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
  inline auto
  perform_hysteresis_and_grouping(ImageView<std::uint8_t>& edges,
                                  const ImageView<float>& orientations,
                                  float angular_threshold)
  {
    const auto index = [&edges](const Eigen::Vector2i& p) {
      return p.y() * edges.width() + p.x();
    };

    const auto is_strong_edgel = [&edges](const Eigen::Vector2i& p) {
      return edges(p) == 255;
    };

    const auto is_weak_edgel = [&edges](const Eigen::Vector2i& p) {
      return edges(p) == 128;
    };

    const auto orientation_vector = [&orientations](const Vector2i& p) {
      const auto& o = orientations(p);
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
        if (is_strong_edgel({x, y}))
          q.emplace(x, y);
      }
    }

    // Neighborhood defined by 8-connectivity.
    const auto dir = std::array<Eigen::Vector2i, 8>{
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
          edges(n) = 255;  // Promote to strong edgel!
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
        if (is_strong_edgel(p))
          contours[static_cast<int>(ds.component(index_p))].push_back(p);
      }
    }

    return contours;
  }

  //  //! @brief Group edgels into **unordered** quasi-straight curves.
  //  inline auto
  //  perform_hysteresis_and_grouping_v2(ImageView<std::uint8_t>& edges,
  //                                     const ImageView<float>& orientations,
  //                                     float angular_threshold)
  //  {
  //    const auto index = [&edges](const Eigen::Vector2i& p) {
  //      return p.y() * edges.width() + p.x();
  //    };

  //    const auto is_edgel = [&edges](const Eigen::Vector2i& p) {
  //      return edges(p) != 0;
  //    };

  //    const auto orientation_vector = [&orientations](const Vector2i& p) {
  //      const auto& o = orientations(p);
  //      return Eigen::Vector2f{cos(o), sin(o)};
  //    };

  //    const auto cosine_similarity = [](const Vector2f& a, const Vector2f& b)
  //    {
  //      return a.dot(b);
  //    };

  //    const auto cosine_angle_threshold = std::cos(angular_threshold);

  //    const auto angle_consistent = [&](const Vector2i& a, const Vector2i& b)
  //    {
  //      const auto oria = orientation_vector(a);
  //      const auto orib = orientation_vector(b);
  //      return cosine_similarity(oria, orib) > cosine_angle_threshold;
  //    };

  //    const auto w = edges.width();
  //    const auto h = edges.height();

  //    // Initialize labeling.
  //    auto labels = Image<std::int32_t>{edges.sizes()};
  //    auto edgels = std::vector<Eigen::Vector2i>{};
  //    for (auto y = 0; y < h; ++y)
  //      for (auto x = 0; x < w; ++x)
  //        labels(x, y) = is_edgel({x, y}) ? y * w + x : -1;

  //    auto ds = DisjointSets(edges.size());

  //    // First pass.
  //  #pragma omp parallel for
  //    for (auto i = 0; i < static_cast<std::int32_t>(edgels.size()); ++i)
  //    {
  //      // e = (x, y) is the current pixel.
  //      const auto& e = edgels[i];
  //      const auto& x = e.x();
  //      const auto& y = e.y();

  //      // Implement the decision tree in Figure 2(b) in the paper.

  //      // b = (x, y - 1) is the north pixel.
  //      if (y > 0 &&                                     //
  //          is_edgel({x, y - 1}) && is_edgel({x, y}) &&  //
  //          angle_consistent({x, y - 1}, {x, y}))
  //        labels(x, y) = labels(x, y - 1);

  //      // c = (x + 1, y - 1) is the north-east pixel.
  //      else if (x + 1 < w && y > 0 &&                            //
  //               is_edgel({x + 1, y - 1}) && is_edgel({x, y}) &&  //
  //               angle_consistent({x + 1, y - 1}, {x, y}))
  //      {
  //        const auto c = labels(x + 1, y - 1);
  //        labels(x, y) = c;

  //        // a = (x - 1, y - 1) is the north-west pixel.
  //        if (x > 0 &&                                         //
  //            is_edgel({x - 1, y - 1}) && is_edgel({x, y}) &&  //
  //            angle_consistent({x - 1, y - 1}, {x, y}))
  //        {
  //          const auto a = labels(x - 1, y - 1);
  //          labels(x, y) = a;
  //        }

  //        else if (x > 0 && labels(x - 1, y) == labels(x, y))
  //        {
  //          const auto d = labels(x - 1, y);
  //          ds.join(ds.node(c), ds.node(d));
  //        }
  //      }

  //      else if (x > 0 && y > 0 && labels(x - 1, y - 1) == labels(x, y))
  //        labels(x, y) = labels(x - 1, y - 1);

  //      else if (x > 0 && labels(x - 1, y) == labels(x, y))
  //        labels(x, y) = labels(x - 1, y);

  //      else
  //      {
  //        ds.make_set(last_label_id);
  //        labels(x, y) = last_label_id;
  //        ++last_label_id;
  //      }
  //    }

  //    // Second pass.
  //    for (int y = 0; y < values.height(); ++y)
  //      for (int x = 0; x < values.width(); ++x)
  //        labels(x, y) = int(ds.component(labels(x, y)));

  //    auto contours = std::map<int, std::vector<Point2i>>{};
  //    for (auto y = 0; y < edges.height(); ++y)
  //    {
  //      for (auto x = 0; x < edges.width(); ++x)
  //      {
  //        const auto p = Eigen::Vector2i{x, y};
  //        const auto index_p = index(p);
  //        if (is_strong_edgel(p))
  //          contours[static_cast<int>(ds.component(index_p))].push_back(p);
  //      }
  //    }

  //    return contours;
  //  }

}  // namespace DO::Sara
