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

#include <drafts/ImageProcessing/EdgeShapeStatistics.hpp>

#include <DO/Sara/Core/PhysicalQuantities.hpp>

#include <DO/Sara/DisjointSets.hpp>

#include <DO/Sara/Geometry/Algorithms/Polyline.hpp>
#include <DO/Sara/Geometry/Tools/Projective.hpp>


namespace DO::Sara {

  // ==========================================================================
  // Edge Grouping By Alignment
  // ==========================================================================
  using Edge = std::vector<Eigen::Vector2d>;

  struct EdgeAttributes
  {
    const std::vector<Edge>& edges;
    const std::vector<Eigen::Vector2d>& centers;
    const std::vector<Eigen::Matrix2d>& axes;
    const std::vector<Eigen::Vector2d>& lengths;
  };

  struct EndPointGraph
  {
    const EdgeAttributes& edge_attrs;

    // List of end points.
    std::vector<Point2d> endpoints;
    // Edge IDs to which the end point belongs to.
    std::vector<std::size_t> edge_ids;

    // Connect the end point to another point.
    // Cannot be in the same edge ids.
    Eigen::MatrixXd score;


    EndPointGraph(const EdgeAttributes& attrs)
      : edge_attrs{attrs}
    {
      endpoints.reserve(2 * edge_attrs.edges.size());
      edge_ids.reserve(2 * edge_attrs.edges.size());
      for (auto i = 0u; i < edge_attrs.edges.size(); ++i)
      {
        const auto& e = edge_attrs.edges[i];
        if (e.size() < 2)
          continue;

        if (length(e) < 5)
          continue;

        const auto& theta = std::abs(std::atan2(edge_attrs.axes[i](1, 0),  //
                                                edge_attrs.axes[i](0, 0)));
        if (theta < 5._deg || std::abs(M_PI - theta) < 5._deg)
          continue;

        endpoints.emplace_back(e.front());
        endpoints.emplace_back(e.back());

        edge_ids.emplace_back(i);
        edge_ids.emplace_back(i);
      }

      score = Eigen::MatrixXd(endpoints.size(), endpoints.size());
      score.fill(std::numeric_limits<double>::infinity());
    }

    auto edge(std::size_t i) const -> const Edge&
    {
      const auto& edge_id = edge_ids[i];
      return edge_attrs.edges[edge_id];
    }

    auto oriented_box(std::size_t i) const -> OrientedBox
    {
      const auto& edge_id = edge_ids[i];
      return {
          edge_attrs.centers[edge_id],  //
          edge_attrs.axes[edge_id],     //
          edge_attrs.lengths[edge_id]   //
      };
    }

    auto mark_plausible_alignments() -> void
    {
      // Tolerance of X degrees in the alignment error.
      const auto thres = std::cos(20._deg);

      for (auto i = 0u; i < endpoints.size() / 2; ++i)
      {
        for (auto k = 0; k < 2; ++k)
        {
          const auto& ik = 2 * i + k;
          const auto& p_ik = endpoints[ik];

          const auto& r_ik = oriented_box(ik);
          const auto& c_ik = r_ik.center;

          for (auto j = i + 1; j < endpoints.size() / 2; ++j)
          {
            // The closest and most collinear point.
            for (int l = 0; l < 2; ++l)
            {
              const auto& jl = 2 * j + l;
              const auto& p_jl = endpoints[jl];

              const auto& r_jl = oriented_box(jl);
              const auto& c_jl = r_jl.center;

              // Particular case:
              if ((p_ik - p_jl).squaredNorm() < 1e-3)
              {
                // Check the plausibility that the end points are mutually
                // aligned?
                const auto dir = std::array<Eigen::Vector2d, 2>{
                    (p_ik - c_ik).normalized(),  //
                    (c_jl - p_jl).normalized()   //
                };

                const auto cosine = dir[0].dot(dir[1]);
                if (cosine < thres)
                  continue;

                score(ik, jl) = 0;
              }
              else
              {
                const auto dir = std::array<Eigen::Vector2d, 3>{
                    (p_ik - c_ik).normalized(),  //
                    (p_jl - p_ik).normalized(),  //
                    (c_jl - p_jl).normalized()   //
                };
                const auto cosines = std::array<double, 2>{
                    dir[0].dot(dir[1]),
                    dir[1].dot(dir[2]),
                };

                if (cosines[0] + cosines[1] < 2 * thres)
                  continue;

                if (Projective::point_to_line_distance(
                        p_ik.homogeneous().eval(), r_jl.line()) > 10 &&
                    Projective::point_to_line_distance(
                        p_jl.homogeneous().eval(), r_ik.line()) > 10)
                  continue;

                // We need this to be as small as possible.
                const auto dist = (p_ik - p_jl).norm();

                // We really need to avoid accidental connections like these
                // situations. Too small edges and too far away, there is little
                // chance it would correspond to a plausible alignment.
                if (length(edge(ik)) + length(edge(jl)) < 20 && dist > 20)
                  continue;

                if (dist > 50)
                  continue;

                score(ik, jl) = dist;
              }
            }
          }
        }
      }
    }

    auto group() const
    {
      const auto n = score.rows();

      auto ds = DisjointSets(n);

      for (auto i = 0; i < n; ++i)
        ds.make_set(i);

      for (auto i = 0; i < n; ++i)
        for (auto j = i; j < n; ++j)
          if (score(i, j) != std::numeric_limits<double>::infinity())
            ds.join(ds.node(i), ds.node(j));

      auto groups = std::map<std::size_t, std::vector<std::size_t>>{};
      for (auto i = 0; i < n; ++i)
      {
        const auto c = ds.component(i);
        groups[c].push_back(i);
      }

      return groups;
    }
  };

}  // namespace DO::Sara
