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

#include <DO/Graphics.hpp>
#include <DO/Geometry.hpp>

namespace DO {
  
  void drawLineSegment(const LineSegment& s, const Color3ub& c, int penWidth)
  {
    drawLine(s.p1(), s.p2(), c, penWidth);
  }

  void drawBBox(const BBox& b, const Color3ub& color, int penWidth)
  {
    Point2i tl(b.top_left().cast<int>());
    Point2i br(b.bottom_right().cast<int>());
    Point2i sz(br-tl);
    drawRect(tl.x(), tl.y(), sz.x(), sz.y(), color, penWidth);
  }

  void drawPoly(const std::vector<Point2d>& p, const Color3ub& color,
                int penWidth)
  {
    for (size_t i1 = 0, i2 = p.size()-1; i1 != p.size(); i2=i1++)
      drawLine(p[i1], p[i2], color, penWidth);
  }

  void drawEllipse(const Ellipse& e, const Color3ub col, int penWidth)
  {
    // Ellipse...
    drawEllipse(e.center(), e.radius1(), e.radius2(),
                to_degree(e.orientation()), col, penWidth);
    // Arrow...
    Vector2d u(unit_vector2(e.orientation()));
    u *= e.radius1()*1.1;
    Point2d a, b;
    a = e.center();
    b = e.center() + u;
    drawArrow(a, b, col, penWidth);
    // Center...
    drawCircle(e.center(), 5., col, penWidth);
  }

  void drawAffineCone(const AffineCone2& K, double arrowLength,
                      const Color3ub& color)
  {
    const Point2d& v = K.vertex();
    Point2d a, b;
    a = v + arrowLength*K.alpha();
    b = v + arrowLength*K.beta();
    drawArrow(v, a, color);
    drawArrow(v, b, color);
  }


} /* namespace DO */