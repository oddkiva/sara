// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
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

inline auto to_map(const std::map<int, std::vector<Eigen::Vector2i>>& contours,
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

inline auto random_colors(const std::map<int, std::vector<Eigen::Vector2i>>& contours)
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
    auto svd = Eigen::BDCSVD<MatrixXf>{inlier_coords, Eigen::ComputeFullU |
                                                          Eigen::ComputeFullV};
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


struct LineSegmentDetector {
  //! @brief intermediate data.
  struct Pipeline {
    Image<Vector2f> gradient_cartesian;
    Image<float> gradient_magnitude;
    Image<float> gradient_orientation;
    Image<std::uint8_t> edge_map;
    std::map<int, std::vector<Point2i>> contours;
    std::vector<std::vector<Eigen::Vector2i>> curve_list;
    std::vector<int> curve_ids;
    std::vector<std::tuple<bool, LineSegment>> line_segments;
  } pipeline;

  struct Parameters {
    //! @brief Canny edge parameters.
    //! @{
    float high_threshold_ratio = 5e-2f;
    float low_threshold_ratio = 2e-2f;
    //! @}

    //! @brief Angle tolerance for connected edgel grouping.
    float angular_threshold = 20. / 180.f * M_PI;

    //! @brief RANSAC-based parameters for line segment fitting.
    //! @{
    int num_iteration_min = 5;
    int num_iteration_max = 20;
    float iteration_percentage = 0.20f;
    //! @}
  } parameters;

  auto operator()(const Image<float>& image)
  {
    pipeline.gradient_cartesian = gradient(image);
    pipeline.gradient_magnitude = pipeline.gradient_cartesian.cwise_transform(
        [](const auto& v) { return v.norm(); });
    pipeline.gradient_orientation = pipeline.gradient_cartesian.cwise_transform(
        [](const auto& v) { return std::atan2(v.y(), v.x()); });

    const auto& grad_mag_max = pipeline.gradient_magnitude.flat_array().maxCoeff();
    const auto& high_thres = grad_mag_max * parameters.high_threshold_ratio;
    const auto& low_thres = grad_mag_max * parameters.low_threshold_ratio;

    pipeline.edge_map = suppress_non_maximum_edgels(  //
        pipeline.gradient_magnitude,                  //
        pipeline.gradient_orientation,                //
        high_thres, low_thres);
    hysteresis(pipeline.edge_map);

    // Extract quasi-straight curve.
    pipeline.contours = connected_components(  //
        pipeline.edge_map,                     //
        pipeline.gradient_orientation,         //
        parameters.angular_threshold);
    const auto labeled_curves = to_map(pipeline.contours, pipeline.edge_map.sizes());

    pipeline.curve_list.clear();
    pipeline.curve_ids.clear();
    for (const auto& [id, curve] : pipeline.contours)
    {
      pipeline.curve_list.emplace_back(curve);
      pipeline.curve_ids.emplace_back(id);
    }

    // Fit a line to each curve.
    pipeline.line_segments = std::vector<std::tuple<bool, LineSegment>>(  //
        pipeline.curve_list.size(),                                       //
        {false, {}}                                                       //
    );
#pragma omp parallel for
    for (auto i = 0; i < static_cast<int>(pipeline.curve_list.size()); ++i)
    {
      const auto& curve = pipeline.curve_list[i];
      if (curve.size() < 5)
        continue;

      auto num_iterations = static_cast<int>(curve.size() * parameters.iteration_percentage) + 1;
      num_iterations = std::clamp(       //
          num_iterations,                //
          parameters.num_iteration_min,  //
          parameters.num_iteration_max);
      pipeline.line_segments[i] = fit_line_segment(  //
          curve,                                     //
          num_iterations,                            //
          /* polish */ true,                        //
          /* line_fit_thresh */ 1.f);                //
    }
  }
};


}  // namespace DO::Sara
