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

#include <DO/Sara/Geometry/Tools/Utilities.hpp>
#include <DO/Sara/Geometry/Algorithms/ConvexHull.hpp>


using namespace std;


namespace DO { namespace Sara { namespace Detail {

  static
  void sort_points_by_polar_angle(PtCotg *out, const Point2d *in,
                                  size_t num_points)
  {
    // Copy.
    for (size_t i = 0; i < num_points; ++i)
      out[i].first = in[i];

    // Find origin and swap with first element.
    PtCotg *origin;
    origin = min_element(out, out + num_points, [](const PtCotg& p, const PtCotg& q)
    {
      if (p.first.y() < q.first.y())
        return true;
      if (p.first.y() == q.first.y() && p.first.x() < q.first.x())
        return true;
      return false;
    });

    swap(*origin, *out);
    // Compute the polar angle w.r.t. origin and sort by polar angle.
    out[0].second = numeric_limits<double>::infinity();
    for (size_t i = 1; i < num_points; ++i)
    {
      Vector2d diff(out[i].first - out[0].first);
      out[i].second = diff.x()/diff.y();
    }
    // Compute the polar angle w.r.t. origin and sort by polar angle.
    sort(out, out+num_points, [](const pair<Point2d, double>& p,
                                const pair<Point2d, double>& q)
    {
      return p.second > q.second;
    });

  }

  void sort_points_by_polar_angle(Point2d *inout, PtCotg *work,
                                  size_t num_points)
  {
    sort_points_by_polar_angle(work, inout, num_points);
    for (size_t i = 0; i < num_points; ++i)
      inout[i] = work[i].first;
  }

} /* namespace Detail */
} /* namespace Sara */
} /* namespace DO */


namespace DO { namespace Sara {

  vector<Point2d> graham_scan_convex_hull(const vector<Point2d>& points)
  {
    using namespace Detail;

    // Sanity check.
    if (points.size() < 3)
      return points;

    // Sort by polar angle.
    vector<PtCotg> point_cotangents_pairs(points.size());
    sort_points_by_polar_angle(&point_cotangents_pairs[0], &points[0], points.size());

    // Weed out the points inside the convex hull.
    std::vector<Point2d> convex_hull;
    convex_hull.reserve(points.size());
    convex_hull.push_back(point_cotangents_pairs[0].first);
    convex_hull.push_back(point_cotangents_pairs[1].first);
    for (size_t i = 2; i != point_cotangents_pairs.size(); ++i)
    {
      while (ccw(convex_hull[convex_hull.size()-2],
                 convex_hull[convex_hull.size()-1],
                 point_cotangents_pairs[i].first) <= 0)
        convex_hull.pop_back();
      convex_hull.push_back(point_cotangents_pairs[i].first);
    }

    return convex_hull;
  }

} /* namespace Sara */
} /* namespace DO */
