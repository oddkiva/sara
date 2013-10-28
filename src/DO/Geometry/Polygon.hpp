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

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

namespace DO {

  // ========================================================================== //
  // Polygon-based approximate computation of the area of intersecting ellipses.
  // ========================================================================== //
  typedef boost::geometry::model::d2::point_xy<double> Point;
  typedef boost::geometry::model::polygon<Point, false> CCWPolygon;

  CCWPolygon discretizeEllipse(const Ellipse& e, int n);

  inline Point2d P(const Point& p)
  { return Point2d(p.x(), p.y()); }

  void drawPolygon(const CCWPolygon& poly, const Rgb8& color);

} /* namespace DO */