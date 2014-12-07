// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Geometry/Tools/Utilities.hpp>
#include <DO/Geometry/Algorithms/ConvexHull.hpp>


using namespace std;


namespace DO { namespace Detail {
    
  static
  inline bool compare_y_coord(const PtCotg& p, const PtCotg& q)
  {
    if (p.first.y() < q.first.y())
      return true;
    if (p.first.y() == q.first.y() && p.first.x() < q.first.x())
      return true;
    return false;
  }

  static
  inline bool compare_cotan(const pair<Point2d, double>& p,
                            const pair<Point2d, double>& q)
  {
    return p.second > q.second;
  }

  static
  void sort_points_by_polar_angle(PtCotg *out, const Point2d *in, int numPoints)
  {
    // Copy.
    for (int i = 0; i < numPoints; ++i)
      out[i].first = in[i];
    // Find origin and swap with first element.
    PtCotg *origin;
    origin = min_element(out, out+numPoints, compare_y_coord);
    swap(*origin, *out);
    // Compute the polar angle w.r.t. origin and sort by polar angle.
    out[0].second = numeric_limits<double>::infinity();
    for (int i = 1; i < numPoints; ++i)
    {
      Vector2d diff(out[i].first - out[0].first);
      out[i].second = diff.x()/diff.y();
    }
    // Compute the polar angle w.r.t. origin and sort by polar angle.
    sort(out, out+numPoints, compare_cotan);
  }

  void sort_points_by_polar_angle(Point2d *inout, PtCotg *work, int numPoints)
  {
    sort_points_by_polar_angle(work, inout, numPoints);
    for (int i = 0; i < numPoints; ++i)
      inout[i] = work[i].first;
  }

} /* namespace internal */
} /* namespace DO */


namespace DO {

  vector<Point2d> graham_scan_convex_hull(const vector<Point2d>& points)
  {
    using namespace Detail;
    // Sanity check.
    if (points.size() < 3)
      return points;
    using namespace Detail;
    // Sort by polar angle.
    vector<PtCotg> ptCotgs(points.size());
    sort_points_by_polar_angle(&ptCotgs[0], &points[0], points.size());
    // Weed out the points inside the convex hull.
    std::vector<Point2d> ch;
    ch.reserve(points.size());
    ch.push_back(ptCotgs[0].first);
    ch.push_back(ptCotgs[1].first);
    for (size_t i = 2; i != ptCotgs.size(); ++i)
    {
      while (ccw(ch[ch.size()-2], ch[ch.size()-1], ptCotgs[i].first) <= 0)
        ch.pop_back();
      ch.push_back(ptCotgs[i].first);
    }

    return ch;
  }

} /* namespace DO */
