// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Geometry/Algorithms/ConvexHull.hpp>
#include <DO/Sara/Geometry/Tools/Utilities.hpp>

#include <iostream>


namespace DO::Sara::Detail {

  static void sort_points_by_polar_angle(PtCotg* out, const Point2d* in,
                                         std::size_t num_points)
  {
    // Copy.
    std::transform(in, in + num_points, out, [](const auto& val) -> PtCotg {
      return {val, 0};
    });

    // Find origin and swap with first element.
    auto origin = std::min_element(
        out, out + num_points, [](const PtCotg& p, const PtCotg& q) {
          return (p.first.y() < q.first.y()) ||
                 (p.first.y() == q.first.y() && p.first.x() < q.first.x());
        });

    std::swap(*origin, *out);

    // Compute the polar angle w.r.t. origin and sort by polar angle.
    std::for_each(out + 1, out + num_points, [out](auto& pt_cotg) {
      // This is slow because of arctan2, but the algorithm is correct.
      const auto diff = Eigen::Vector2d(pt_cotg.first - out->first);
      if (diff.y() == 0)
        pt_cotg.second =
            std::copysign(std::numeric_limits<double>::max(), diff.x());
      else
        pt_cotg.second = diff.x() / diff.y();
    });
    // Compute the polar angle w.r.t. origin and sort by polar angle.
    std::sort(out + 1, out + num_points, [](const PtCotg& p, const PtCotg& q) {
      return p.second > q.second;
    });
  }

  void sort_points_by_polar_angle(Point2d* inout, PtCotg* work,
                                  size_t num_points)
  {
    sort_points_by_polar_angle(work, inout, num_points);
    std::transform(work, work + num_points, inout,
                   [](const auto& val) { return val.first; });
  }

}  // namespace DO::Sara::Detail


namespace DO::Sara {

  auto graham_scan_convex_hull(const std::vector<Point2d>& points)
      -> std::vector<Point2d>
  {
    using namespace Detail;

    // Sanity check.
    if (points.size() < 3)
      return points;

    // Sort by polar angle.
    auto point_cotangents_pairs = std::vector<PtCotg>(points.size());
    sort_points_by_polar_angle(&point_cotangents_pairs[0], &points[0],
                               points.size());

    // Keep the farthest point if there is a cluster of points with the same
    // polar angle.
    {
      auto point_cotangents_pairs_filtered = std::vector<PtCotg>{};
      point_cotangents_pairs_filtered.reserve(point_cotangents_pairs.size());

      const auto& origin = point_cotangents_pairs.front().first;
      point_cotangents_pairs_filtered.push_back(point_cotangents_pairs.front());

      auto a = point_cotangents_pairs.begin() + 1;
      do
      {
        // Calculate the open interval [a, b)
        auto b = a + 1;
        while (b != point_cotangents_pairs.end() && b->second == a->second)
          ++b;

        // Calculate the farthest point from the origin.
        const auto x = std::max_element(  //
            a, b,                         //
            [origin](const PtCotg& p, const PtCotg& q) {
              return (p.first - origin).squaredNorm() <
                     (q.first - origin).squaredNorm();
            });
        point_cotangents_pairs_filtered.push_back(*x);

        // Go to the next interval.
        a = b;
      } while (a != point_cotangents_pairs.end());

      point_cotangents_pairs.swap(point_cotangents_pairs_filtered);
    }

    // Weed out the points inside the convex hull.

    auto convex_hull = std::vector<Point2d>{};
    convex_hull.reserve(points.size());
    convex_hull.push_back(point_cotangents_pairs[0].first);
    convex_hull.push_back(point_cotangents_pairs[1].first);
    for (std::size_t i = 2; i != point_cotangents_pairs.size(); ++i)
    {
      while (ccw(*(convex_hull.rbegin() + 1), *(convex_hull.rbegin()),
                 point_cotangents_pairs[i].first) <= 0)
        convex_hull.pop_back();
      convex_hull.push_back(point_cotangents_pairs[i].first);
    }

    return convex_hull;
  }

}  // namespace DO::Sara
