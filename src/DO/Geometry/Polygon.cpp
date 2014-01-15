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

#include <DO/Geometry.hpp>

using namespace std;

namespace DO {
  
  inline
  bool compareY(const Point2d& p, const Point2d& q)
  {
    if (p.y() < q.y())
      return true;
    if (p.y() == q.y() && p.x() < q.x())
      return true;
    return false;
  }
  
  inline
  bool compareLexicographically(const Point2d& p, const Point2d& q)
  {
    return lexicographical_compare(p.data(), p.data()+2, q.data(), q.data()+2);
  }
  
  typedef pair<Point2d, double> PtCot;
  
  inline
  bool compareCotan(const PtCot& p, const PtCot& q)
  {
    return p.second > q.second;
  }

  inline
  double ccw(const Point2d& a, const Point2d& b, const Point2d& c)
  {
    Matrix2d A;
    A.col(0) = (b-a);
    A.col(1) = (c-a);
    return A.determinant();
  }

  vector<Point2d> grahamScanConvexHull(const vector<Point2d>& points)
  {
    // Sanity check.
    if (points.size() < 3)
      return points;

    // Eliminate redundant points.
    vector<Point2d> ch(points);
    //sort(ch.begin(), ch.end(), compareLexicographically);
    //ch.resize(unique(ch.begin(), ch.end()) - ch.begin());
    
    // Sanity check.
    if (ch.size() < 3)
      return ch;

    // Find the point with the lowest y-coordinate value.
    vector<Point2d>::iterator lowestY;
    lowestY = min_element(ch.begin(), ch.end(), compareY);
    swap(*lowestY, ch[0]);
    
    // Sort points by cotangent values.
    vector<PtCot> sortedPts(ch.size());
    sortedPts[0] = make_pair(ch[0], numeric_limits<double>::infinity());
    for (size_t i = 1; i != ch.size(); ++i)
    {
      Vector2d diff(ch[i]-ch[0]);
      sortedPts[i] = make_pair(ch[i], diff.x()/diff.y());
    }
    sort(sortedPts.begin(), sortedPts.end(), compareCotan);
    
    // Weed out the points inside the convex hull.
    ch.resize(2);
    for (size_t i = 2; i != sortedPts.size(); ++i)
    {
      while (ccw(ch[ch.size()-2], ch[ch.size()-1], sortedPts[i].first) <= 0)
        ch.pop_back();
      ch.push_back(sortedPts[i].first);
    }

    return ch;
  }

} /* namespace DO */