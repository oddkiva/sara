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
#include "Polygon.hpp"

using namespace std;
using namespace boost::geometry;

namespace DO {

  // ========================================================================== //
  // Polygon-based approximate computation of the area of intersecting ellipses.
  // ========================================================================== //
  CCWPolygon toCCWPoly(const vector<Point2d>& points)
  {

    CCWPolygon polygon;
    for (size_t i = 0; i != points.size(); ++i)
      append(polygon, Point(points[i].x(), points[i].y()));
    return polygon;
  }

  vector<Point2d> toVector(const CCWPolygon& polygon)
  {
    vector<Point2d> vec;
    for (size_t i = 0; i != polygon.outer().size(); ++i)
    {
      double x = polygon.outer()[i].x();
      double y = polygon.outer()[i].y();
      vec.push_back(Point2d(x,y));
    }
    return vec;
  }
  
  CCWPolygon discretizeEllipse(const Ellipse& e, int n)
  {
    CCWPolygon polygon;

    const Matrix2d Ro(rotation2(e.o()));
    Vector2d D( e.r1(), e.r2() );

    for(int i = 0; i < n; ++i)
    {
      const double theta = 2.*M_PI*double(i)/n;
      const Matrix2d R(rotation2(theta));
      Point2d p(1.0, 0.0);

      const Point2d p1(e.c() + Ro.matrix()*D.asDiagonal()*R.matrix()*p);
      boost::geometry::append(polygon, Point(p1.x(), p1.y()));
    }

    return polygon;
  }

  void drawPolygon(const CCWPolygon& poly, const Rgb8& color)
  {
    using namespace boost::geometry;
    typedef CCWPolygon::ring_type Ring;
    const Ring& ring = exterior_ring(poly);
    for (size_t i = 0; i != ring.size(); ++i)
    {
      Point2f pi(P(ring[i]).cast<float>());
      Point2f pi1(P(ring[(i+1)%ring.size()]).cast<float>());
      fillCircle(pi, 3.f, color);
      fillCircle(pi1, 3.f, color);
      drawLine(pi, pi1, color);
    }
  }

  vector<Point2d> convexHull(const vector<Point2d>& points)
  {

    CCWPolygon polygon(toCCWPoly(points));
    CCWPolygon hull;
    convex_hull(polygon, hull);
    return toVector(hull);
  }

  double convexHullArea(const vector<Point2d>& points)
  {
    CCWPolygon polygon(toCCWPoly(points));
    CCWPolygon hull;
    convex_hull(polygon, hull);
    return area(hull);
  }

} /* namespace DO */