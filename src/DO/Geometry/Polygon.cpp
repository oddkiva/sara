// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Graphics.hpp>
#include <DO/Geometry.hpp>

using namespace std;

namespace DO {

  inline double ccw(const Point2d& a, const Point2d& b, const Point2d& c)
  {
    Matrix2d A;
    A.col(0) = (b-a);
    A.col(1) = (c-a);
    return A.determinant();
  }

  vector<Point2d> grahamScanConvexHull(const vector<Point2d>& points)
  {
    if (points.size() < 3)
      throw std::runtime_error("Cannot compute convex hull: points.size() < 3");

    vector<Point2d> ch(points);

    auto lowestY = min_element(ch.begin(), ch.end(), 
      [](const Point2d& p, const Point2d& q)
      {
        if (p.y() < q.y())
          return true;
        if (p.y() == q.y() && p.x() < q.x())
          return true;
        return false;
      }
    );
    swap(*lowestY, ch[0]);

    vector<Point2d> convexHull;

    struct PtVal {
      PtVal(const Point2d& p_)
        : p(p_)
        , cotg(p_.y() / p_.x())
      {}
      Point2d p;
      double cotg;
      bool operator<(const PtVal& other) const
      {
        if (cotg > other.cotg)
          return true;
        if (cotg == other.cotg && p.x() < other.p.x())
          return true;
        return false;
      }
    };

    vector<PtVal> sortedPts;
    for (const auto p : points)
      sortedPts.push_back(p);
    sort(sortedPts.begin(), sortedPts.end());

    int M = 1;
    for (size_t i = 2; i != sortedPts.size(); ++i)
    {
      while (ccw(sortedPts[i-2].p, sortedPts[i-1].p, sortedPts[i].p) <= 0)
      {
        if (M > 1)
          M--;
        else if (i==N)
          break;
        else
          ++i;
      }
    }


    return convexHull;
  }

} /* namespace DO */