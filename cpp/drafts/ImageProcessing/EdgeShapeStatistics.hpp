// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/TicToc.hpp>

#include <DO/Sara/Geometry/Algorithms/Polyline.hpp>
#include <DO/Sara/Geometry/Objects/LineSegment.hpp>
#include <DO/Sara/Geometry/Tools/Projective.hpp>

#include <DO/Sara/Graphics/ImageDraw.hpp>


namespace DO::Sara {

  // ==========================================================================
  // Shape Statistics of the Edge.
  // ==========================================================================
  struct OrientedBox
  {
    const Eigen::Vector2d& center;
    const Eigen::Matrix2d& axes;
    const Eigen::Vector2d& lengths;

    auto line() const
    {
      return Projective::line(center.homogeneous().eval(),
                              (center + axes.col(0)).homogeneous().eval());
    }

    auto draw(ImageView<Rgb8>& detection, const Rgb8& color,  //
              const Point2d& c1, const double s) const -> void
    {
      const Vector2d u = axes.col(0);
      const Vector2d v = axes.col(1);
      const auto p = std::array<Vector2d, 4>{
          c1 + s * (center + (lengths(0) + 0) * u + (lengths(1) + 0) * v),
          c1 + s * (center - (lengths(0) + 0) * u + (lengths(1) + 0) * v),
          c1 + s * (center - (lengths(0) + 0) * u - (lengths(1) + 0) * v),
          c1 + s * (center + (lengths(0) + 0) * u - (lengths(1) + 0) * v),
      };
      auto pi = std::array<Vector2i, 4>{};
      std::transform(p.begin(), p.end(), pi.begin(),
                     [](const Vector2d& v) { return v.cast<int>(); });

      draw_line(detection, pi[0].x(), pi[0].y(), pi[1].x(), pi[1].y(), color,
                2);
      draw_line(detection, pi[1].x(), pi[1].y(), pi[2].x(), pi[2].y(), color,
                2);
      draw_line(detection, pi[2].x(), pi[2].y(), pi[3].x(), pi[3].y(), color,
                2);
      draw_line(detection, pi[3].x(), pi[3].y(), pi[0].x(), pi[0].y(), color,
                2);
    }
  };


  struct CurveStatistics
  {
    // TODO: figure out why the linear directional mean is shaky.
    // std::vector<double> ldms;

    // The rectangle approximation.
    std::vector<Vector2d> centers;
    std::vector<Matrix2d> inertias;
    std::vector<Matrix2d> axes;
    std::vector<Vector2d> lengths;

    CurveStatistics() = default;

    CurveStatistics(const std::vector<std::vector<Eigen::Vector2d>>& edges)
    {
      tic();

      // TODO: figure out why the linear directional mean is shaky.
      // auto ldms = std::vector<double>(edges.size());
      // The rectangle approximation.
      centers = std::vector<Vector2d>(edges.size());
      inertias = std::vector<Matrix2d>(edges.size());
      axes = std::vector<Matrix2d>(edges.size());
      lengths = std::vector<Vector2d>(edges.size());

#pragma omp parallel for
      for (auto i = 0; i < static_cast<int>(edges.size()); ++i)
      {
        const auto& e = edges[i];
        if (e.size() < 2)
          continue;

        // ldms[i] = linear_directional_mean(e);
        centers[i] = center_of_mass(e);
        inertias[i] = matrix_of_inertia(e, centers[i]);
        const auto svd = inertias[i].jacobiSvd(Eigen::ComputeFullU);
        axes[i] = svd.matrixU();
        lengths[i] = svd.singularValues().cwiseSqrt();
      }

      toc("Edge Shape Statistics");
    }

    auto swap(CurveStatistics& other)
    {
      centers.swap(other.centers);
      inertias.swap(other.inertias);
      axes.swap(other.axes);
      lengths.swap(other.lengths);
    }

    auto oriented_box(int i) const -> OrientedBox
    {
      return {centers[i], axes[i], lengths[i]};
    }
  };


  auto extract_line_segments_quick_and_dirty(const CurveStatistics& stats,
                                             float thinness_ratio = 5.f)
  {
    auto line_segments = std::vector<LineSegment>{};
    line_segments.reserve(stats.axes.size());

    for (auto i = 0u; i < stats.axes.size(); ++i)
    {
      const auto& box = stats.oriented_box(i);
      const auto is_box_thin = box.lengths(0) > thinness_ratio * box.lengths(1);
      if (!is_box_thin)
        continue;

      const Eigen::Vector2d p1 = box.center - box.axes.col(0) * box.lengths(0);
      const Eigen::Vector2d p2 = box.center + box.axes.col(0) * box.lengths(0);

      line_segments.emplace_back(p1, p2);
    }

    return line_segments;
  }

  auto to_lines(const std::vector<LineSegment>& line_segments)
      -> Tensor_<float, 2>
  {
    auto lines = Tensor_<float, 2>(static_cast<int>(line_segments.size()), 3);
    auto lines_as_matrix = lines.matrix();
    for (auto i = 0u; i < line_segments.size(); ++i)
    {
      const auto& ls = line_segments[i];
      const Eigen::Vector3f p1 = ls.p1().homogeneous().cast<float>();
      const Eigen::Vector3f p2 = ls.p2().homogeneous().cast<float>();
      auto line = Projective::line(p1, p2);
      line /= line.head(2).norm();
      lines_as_matrix.row(i) = line.transpose();
    }

    return lines;
  }

}  // namespace DO::Sara
