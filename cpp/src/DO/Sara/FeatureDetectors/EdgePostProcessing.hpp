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
#include <DO/Sara/Geometry/Objects/LineSegment.hpp>
#include <DO/Sara/RANSAC/RANSAC.hpp>


namespace DO::Sara {

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


  template <typename T>
  inline auto
  fit_line_segment(const std::vector<Eigen::Matrix<T, 2, 1>>& points)
      -> LineSegment
  {
    if (points.size() < 2)
      throw std::runtime_error{"Invalid polyline!"};

    // Optimization.
    if (points.size() == 2)
      return {points[0].template cast<double>(),
              points[1].template cast<double>()};

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

    const auto num_curve_points = static_cast<int>(curve_points.size());
    auto point_list = PointList<float, 2>{};
    point_list._data.resize(num_curve_points, 3);
    auto& points = point_list._data;
    auto point_matrix = points.matrix();
    for (auto r = 0; r < num_curve_points; ++r)
      point_matrix.row(r) = curve_points[r]      //
                                .transpose()     //
                                .homogeneous()   //
                                .cast<float>();  //

    auto line_solver = LineSolver2D<float>{};
    auto inlier_predicate = InlierPredicate{
        LinePointDistance2D<float>{},  //
        error_threshold                //
    };
    const auto ransac_result = ransac_v2(point_list,        //
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
