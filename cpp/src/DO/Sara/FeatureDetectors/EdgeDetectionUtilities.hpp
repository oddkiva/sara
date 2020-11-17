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

#include <DO/Sara/ImageProcessing.hpp>

#include <DO/Sara/Geometry/Algorithms/RobustEstimation/LineSolver.hpp>
#include <DO/Sara/Geometry/Algorithms/RobustEstimation/RANSAC.hpp>
#include <DO/Sara/Geometry/Objects/LineSegment.hpp>


namespace DO::Sara {

  inline auto
  to_map(const std::map<int, std::vector<Eigen::Vector2i>>& contours,
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

  inline auto
  fit_line_segment_robustly(const std::vector<Eigen::Vector2i>& curve_points,
                            int num_iterations,   //
                            bool polish = false,  //
                            float error_threshold = 1.f,
                            float min_consensus_ratio = 0.5f)
      -> std::tuple<bool, LineSegment>
  {
    enum class Axis : std::uint8_t
    {
      X = 0,
      Y = 1
    };

    if (curve_points.size() < 2)
      return {false, {}};

    auto line_solver = LineSolver2D<float>{};
    auto inlier_predicate = InlierPredicate<LinePointDistance2D<float>>{
        {},              //
        error_threshold  //
    };

    auto points = Tensor_<float, 2>(curve_points.size(), 3);
    auto point_matrix = points.matrix();
    for (auto r = 0u; r < curve_points.size(); ++r)
      point_matrix.row(r) = curve_points[r]      //
                                .transpose()     //
                                .homogeneous()   //
                                .cast<float>();  //

    const auto ransac_result = ransac(points,            //
                                      line_solver,       //
                                      inlier_predicate,  //
                                      num_iterations);
    const auto& line = std::get<0>(ransac_result);
    const auto& inliers = std::get<1>(ransac_result);

    // Do we have sufficiently enough inliers?
    const auto inlier_count = inliers.flat_array().count();
    if (inlier_count < min_consensus_ratio * curve_points.size())
      return {false, {}};

    auto inlier_coords = MatrixXf{inlier_count, 3};
    for (auto i = 0, j = 0; i < point_matrix.rows(); ++i)
    {
      if (!inliers(i))
        continue;

      inlier_coords.row(j) = point_matrix.row(i);
      ++j;
    }

    Eigen::Vector2f t = Projective::tangent(line).cwiseAbs();
    auto longest_axis = t.x() > t.y() ? Axis::X : Axis::Y;

    auto min_index = 0;
    auto max_index = 0;
    if (longest_axis == Axis::X)
    {
      inlier_coords.col(0).minCoeff(&min_index);
      inlier_coords.col(0).maxCoeff(&max_index);
    }
    else
    {
      inlier_coords.col(1).minCoeff(&min_index);
      inlier_coords.col(1).maxCoeff(&max_index);
    }
    Eigen::Vector2f tl = inlier_coords.row(min_index).hnormalized().transpose();
    Eigen::Vector2f br = inlier_coords.row(max_index).hnormalized().transpose();

    // Polish the line segment.
    if (polish && inlier_count > 3)
    {
      auto svd = Eigen::BDCSVD<MatrixXf>{
          inlier_coords, Eigen::ComputeFullU | Eigen::ComputeFullV};
      const Eigen::Vector3f l = svd.matrixV().col(2);

      t = Projective::tangent(l).cwiseAbs();
      longest_axis = t.x() > t.y() ? Axis::X : Axis::Y;

      if (longest_axis == Axis::X)
      {
        tl.y() = -(l(0) * tl.x() + l(2)) / l(1);
        br.y() = -(l(0) * br.x() + l(2)) / l(1);
      }
      else
      {
        tl.x() = -(l(1) * tl.y() + l(2)) / l(0);
        br.x() = -(l(1) * br.y() + l(2)) / l(0);
      }
    }

    return {true, {tl.cast<double>(), br.cast<double>()}};
  }


  // ======================================================================== //
  // Algorithmic building blocks.
  // ==========================================================================
  auto residual(const ImageView<float>& g, const Eigen::Vector2i& x,
                float eps = 1e-6f) -> Eigen::Vector2f
  {
    const auto g1 = gradient(g, x);
    const auto g2 = hessian(g, x);

    if (std::abs(g2.determinant()) < eps)
      return Eigen::Vector2f::Zero();

    const Eigen::Vector2f r = -g2.inverse() * g1;
    return r.cwiseAbs().maxCoeff() >= 1 ? Eigen::Vector2f::Zero() : r;
  }

  auto refine(const ImageView<float>& g, const Eigen::Vector2i& x,
              float eps = 1e-6f) -> Eigen::Vector2f
  {
    return x.cast<float>() + residual(g, x, eps);
  }

  //! @brief Specifically calcuates the linear directional mean of the polyline
  //! `p` reweighted by the length of line segment.
  template <typename Point>
  auto linear_directional_mean(const std::vector<Point>& p)
  {
    const auto dirs = std::vector<Eigen::Vector2f>(p.size() - 1);
    std::adjacent_difference(p.begin(), p.end(), dirs.begin());

    const auto cosine = std::accumulate(
        dirs.begin(), dirs.end(), float{},
        [](const auto& sum, const auto& d) { return sum + d.x(); });
    const auto sine = std::accumulate(
        dirs.begin(), dirs.end(), float{},
        [](const auto& sum, const auto& d) { return sum + d.y(); });

    return std::atan2(sine, cos);
  }


  // template <typename Point>
  // auto collapse(const std::vector<Point>& p,
  //               const float length_threshold = 5.e-2f)
  //     -> std::vector<Point>
  // {
  //   auto deltas = std::vector<Eigen::Vector2f>(p.size() - 1);
  //   std::adjacent_difference(p.begin(), p.end(), deltas.begin());
  //
  //   auto lengths = std::vector<float>(deltas.size());
  //   std::transform(deltas.begin(), deltas.end(), lengths.begin(),
  //                  [](const auto& d) { return d.norm(); });
  //
  //   // Normalize.
  //   const auto total_length = std::accumulate(lengths.begin(), lengths.end(),
  //   float{}); std::for_each(lengths.begin(), lengths.end(), lengths.begin(),
  //                 [&](auto& l) { l /= total_length; });
  //
  //   // Find the cuts.
  //   auto collapse_state = std::vector<std::uint8_t>(p.size(), 0);
  //   for (auto i = 0; i < p.size() - 1; ++i)
  //   {
  //     if (lengths[i] < length_threshold)
  //     {
  //       collapse_state[i] = 1;
  //       collapse_state[i + 1] = 1;
  //     }
  //   }
  //
  //   auto p_collapsed = std::vector<Point>{};
  //   for (auto i = 0; i < p.size();)
  //   {
  //     if (collapse_state[i] == 0)
  //     {
  //       ++i;
  //       continue;
  //     }
  //
  //     auto a = p.begin() + i;
  //     auto b = std::find(a, p.end(), 0);

  //     const auto cardinality = b - a;

  //     auto sum =
  //         std::accumulate(a, b, Eigen::Vector2f{0, 0},
  //                         [](const auto& a, const auto& b) { return a + b; }) /
  //         cardinality;
  //   }
  //
  //   // FINISH.
  // }


  //! TODO: see if reweighting with edge length improves the split.
  template <typename Point>
  auto split(const std::vector<Point>& ordered_points,
             const float angle_threshold = M_PI / 6)
      -> std::vector<std::vector<Point>>
  {
    if (angle_threshold >= M_PI)
      throw std::runtime_error{"Invalid angle threshold!"};

    if (ordered_points.size() < 3)
      return {ordered_points};

    const auto& p = ordered_points;
    const auto& cos_threshold = std::cos(angle_threshold);

    // Calculate the orientation of line segments.
    auto deltas = std::vector<Point>(p.size() - 1);
    // Subtract.
    std::adjacent_difference(p.begin(), p.end(), deltas.begin());
    // Normalize.
    std::for_each(deltas.begin(), deltas.end(),
                  [](auto& v) { return v.normalized(); });

    // Find the cuts.
    auto cuts = std::vector<std::uint8_t>(p.size(), 0);
    std::adjacent_difference(deltas.begin(), deltas.end(), cuts.begin(),
                             [&](const auto& a, const auto& b) {
                               const auto cosine = a.dot(b);
                               return cosine < cos_threshold;
                             });

    auto pp = std::vector<std::vector<Point>>{};
    pp.push_back({p[0]});
    for (auto i = 1u; i < p.size(); ++i)
    {
      pp.back().push_back(p[i]);
      if (cuts[i] == 1)
        pp.push_back({p[i]});
    }

    return pp;
  }


  template <typename Point>
  auto split(const std::vector<std::vector<Point>>& edges,
             const float angle_threshold = M_PI / 6)
      -> std::vector<std::vector<Point>>
  {
    auto edges_split = std::vector<std::vector<Point>>{};
    edges_split.reserve(2 * edges.size());
    for (const auto& e : edges)
      append(edges_split, split(e, angle_threshold));

    return edges_split;
  }

  inline auto fit_line_segment(const std::vector<Eigen::Vector2i>& points)
      -> LineSegment
  {
    auto coords = MatrixXf{points.size(), 3};
    for (auto i = 0; i < coords.rows(); ++i)
      coords.row(i) = points[i].homogeneous().transpose().cast<float>();

    // Calculate the line equation `l`.
    auto svd = Eigen::BDCSVD<MatrixXf>{coords, Eigen::ComputeFullU |
                                                   Eigen::ComputeFullV};

    const Eigen::Vector3f l = svd.matrixV().col(2);

    // Direction vector.
    const auto t = Projective::tangent(l).cwiseAbs();

    enum class Axis : std::uint8_t
    {
      X = 0,
      Y = 1
    };
    const auto longest_axis = t.x() > t.y() ? Axis::X : Axis::Y;

    Eigen::Vector2f p1 = points.front().cast<float>();
    Eigen::Vector2f p2 = points.back().cast<float>();

    if (longest_axis == Axis::X)
    {
      p1.y() = -(l(0) * p1.x() + l(2)) / l(1);
      p2.y() = -(l(0) * p2.x() + l(2)) / l(1);
    }
    else
    {
      p1.x() = -(l(1) * p1.y() + l(2)) / l(0);
      p2.x() = -(l(1) * p2.y() + l(2)) / l(0);
    }

    return {p1.cast<double>(), p2.cast<double>()};
  }


  // ======================================================================== //
  // Draw.
  // ======================================================================== //
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
