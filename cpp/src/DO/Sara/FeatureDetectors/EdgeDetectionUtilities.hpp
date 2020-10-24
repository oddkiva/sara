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


  inline auto fit_line_segment(const std::vector<Eigen::Vector2i>& curve_points,
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
