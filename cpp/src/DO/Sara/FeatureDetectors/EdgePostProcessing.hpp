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

#include <DO/Sara/ImageProcessing.hpp>

#include <DO/Sara/Geometry/Algorithms/RobustEstimation/LineSolver.hpp>
#include <DO/Sara/Geometry/Algorithms/RobustEstimation/RANSAC.hpp>
#include <DO/Sara/Geometry/Objects/LineSegment.hpp>


namespace DO::Sara {

  // ======================================================================== //
  // Algorithmic Building Blocks for Curve Post-Processing.
  // ==========================================================================

  //! @brief Reorder and extract the longest curve from the point set.
  //! N.B.: this is a heuristic approach but it does work well.
  inline auto reorder_and_extract_longest_curve(
      const std::vector<Eigen::Vector2i>& curve_points,
      int connectivity_threshold = 2) -> std::vector<Eigen::Vector2i>
  {
    enum class Axis : std::uint8_t
    {
      X = 0,
      Y = 1
    };

    if (curve_points.size() <= 2)
      return {};

    const Eigen::Vector2i min = std::accumulate(
        curve_points.begin(), curve_points.end(), curve_points.front(),
        [](const auto& a, const auto& b) { return a.cwiseMin(b); });
    const Eigen::Vector2i max = std::accumulate(
        curve_points.begin(), curve_points.end(), curve_points.front(),
        [](const auto& a, const auto& b) { return a.cwiseMax(b); });
    const Eigen::Vector2i delta = (max - min).cwiseAbs();

    const auto longest_axis = delta.x() > delta.y() ? Axis::X : Axis::Y;

    auto compare_xy = [](const auto& a, const auto& b) {
      if (a.x() < b.x())
        return true;
      if (a.x() == b.x() && a.y() < b.y())
        return true;
      return false;
    };

    auto compare_yx = [](const auto& a, const auto& b) {
      if (a.y() < b.y())
        return true;
      if (a.y() == b.y() && a.x() < b.x())
        return true;
      return false;
    };

    auto curve_points_sorted = curve_points;
    if (longest_axis == Axis::X)
      std::sort(curve_points_sorted.begin(), curve_points_sorted.end(),
                compare_xy);
    else
      std::sort(curve_points_sorted.begin(), curve_points_sorted.end(),
                compare_yx);

    auto curve_points_ordered = std::vector<Eigen::Vector2i>{};
    curve_points_ordered.emplace_back(curve_points_sorted.front());
    for (auto i = 1u; i < curve_points_sorted.size(); ++i)
    {
      if ((curve_points_ordered.back() - curve_points_sorted[i])
              .lpNorm<Eigen::Infinity>() <= connectivity_threshold)
        curve_points_ordered.emplace_back(curve_points_sorted[i]);
    }

    return curve_points_ordered;
  }

  //! @brief Refine the location of each edgel.
  /*!
   *  We reuse the same technique used in SIFT where we fit a second-order
   *  polynomial obtained from a Taylor expansion.
   *  @{
   */
  inline auto residual(const ImageView<float>& g, const Eigen::Vector2i& x,
                       float eps = 1e-6f) -> Eigen::Vector2f
  {
    const auto g1 = gradient(g, x);
    const auto g2 = hessian(g, x);

    if (std::abs(g2.determinant()) < eps)
      return Eigen::Vector2f::Zero();

    const Eigen::Vector2f r = -g2.inverse() * g1;
    return r.cwiseAbs().maxCoeff() >= 1 ? Eigen::Vector2f::Zero() : r;
  }

  inline auto refine(const ImageView<float>& g, const Eigen::Vector2i& x,
                     float eps = 1e-6f) -> Eigen::Vector2f
  {
    return x.cast<float>() + residual(g, x, eps);
  }
  //! @}

  //! @brief Calculate the linear directional mean of the polyline.
  /*!
   *  The linear directional mean is the mean orientation reweighted by the
   *  length of each line segment in the polyline.
   */
  template <typename Point>
  auto linear_directional_mean(const std::vector<Point>& p)
  {
    const auto dirs = std::vector<Eigen::Vector2f>(p.size());
    std::adjacent_difference(p.begin(), p.end(), dirs.begin());

    const auto cosine = std::accumulate(
        dirs.begin(), dirs.end(), float{},
        [](const auto& sum, const auto& d) { return sum + d.x(); });
    const auto sine = std::accumulate(
        dirs.begin(), dirs.end(), float{},
        [](const auto& sum, const auto& d) { return sum + d.y(); });

    return std::atan2(sine, cos);
  }


  template <typename T, int N>
  auto collapse(const std::vector<Eigen::Matrix<T, N, 1>>& p,
                const ImageView<float>& gradient,  //
                T threshold = 5e-2, bool adaptive = true)
      -> std::vector<Eigen::Matrix<T, N, 1>>
  {
    if (p.size() < 2)
      throw std::runtime_error{"Invalid polyline!"};

    using Point = Eigen::Matrix<T, N, 1>;

    auto deltas = std::vector<Point>(p.size() - 1);
    for (auto i = 0u; i < deltas.size(); ++i)
      deltas[i] = p[i + 1] - p[i];

    auto lengths = std::vector<T>(deltas.size());
    std::transform(deltas.begin(), deltas.end(), lengths.begin(),
                   [](const auto& d) { return d.norm(); });

    if (adaptive)
    {
      const auto total_length =
          std::accumulate(lengths.begin(), lengths.end(), T{});
      for (auto& l : lengths)
        l /= total_length;
    }

    // Find the cuts.
    auto collapse_state = std::vector<std::uint8_t>(p.size(), 0);
    for (auto i = 0u; i < lengths.size(); ++i)
    {
      if (lengths[i] < threshold)
      {
        collapse_state[i] = 1;
        collapse_state[i + 1] = 1;
      }
    }

    auto p_collapsed = std::vector<Point>{};
    p_collapsed.reserve(p.size());
    for (auto i = 0u; i < p.size();)
    {
      if (collapse_state[i] == 0)
      {
        p_collapsed.push_back(p[i]);
        ++i;
        continue;
      }

      const auto& a = i;
      const auto b =
          std::find(collapse_state.begin() + a, collapse_state.end(), 0) -
          collapse_state.begin();

      const auto pa = p.begin() + a;
      const auto pb = p.begin() + b;

      const auto best =
          std::max_element(pa, pb, [&gradient](const auto& u, const auto& v) {
            return gradient(u.template cast<int>()) <
                   gradient(v.template cast<int>());
          });
      p_collapsed.emplace_back(*best);

      i = b;
    }

    return p_collapsed;
  }


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
    for (auto i = 0u; i < deltas.size(); ++i)
      deltas[i] = (p[i + 1] - p[i]).normalized();

    // Find the cuts.
    auto cuts = std::vector<std::uint8_t>(p.size(), 0);
    for (auto i = 0u; i < deltas.size() - 1; ++i)
    {
      const auto cosine = deltas[i].dot(deltas[i + 1]);
      // a, b, c
      // a = p[i]
      // b = p[i + 1]
      // c = p[i + 2]
      cuts[i + 1] = cosine < cos_threshold;
    }

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


  template <typename T>
  inline auto
  fit_line_segment(const std::vector<Eigen::Matrix<T, 2, 1>>& points)
      -> LineSegment
  {
    if (points.size() < 2)
      throw std::runtime_error{"Invalid polyline!"};

    // Optimization.
    if (points.size() == 2)
      return {points.front().template cast<double>(),
              points.back().template cast<double>()};

    // General case.
    auto coords = Matrix<T, Eigen::Dynamic, 3>{points.size(), 3};
    for (auto i = 0; i < coords.rows(); ++i)
      coords.row(i) = points[i].homogeneous().transpose();

    // Calculate the line equation `l`.
    auto svd = Eigen::BDCSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>{
        coords, Eigen::ComputeFullU | Eigen::ComputeFullV};

    const Eigen::Matrix<T, 3, 1> l = svd.matrixV().col(2);

    // Direction vector.
    const auto t = Projective::tangent(l).cwiseAbs();

    enum class Axis : std::uint8_t
    {
      X = 0,
      Y = 1
    };
    const auto longest_axis = t.x() > t.y() ? Axis::X : Axis::Y;

    auto min_index = 0;
    auto max_index = 0;
    if (longest_axis == Axis::X)
    {
      coords.col(0).minCoeff(&min_index);
      coords.col(0).maxCoeff(&max_index);
    }
    else
    {
      coords.col(1).minCoeff(&min_index);
      coords.col(1).maxCoeff(&max_index);
    }
    Eigen::Matrix<T, 2, 1> p1 = coords.row(min_index).hnormalized().transpose();
    Eigen::Matrix<T, 2, 1> p2 = coords.row(max_index).hnormalized().transpose();

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

    return {p1.template cast<double>(), p2.template cast<double>()};
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

}  // namespace DO::Sara
