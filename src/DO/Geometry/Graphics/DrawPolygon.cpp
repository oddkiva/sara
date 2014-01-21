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
    Point2i tl(b.topLeft().cast<int>());
    Point2i br(b.bottomRight().cast<int>());
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
    drawEllipse(
      e.c().cast<float>(),
      static_cast<float>(e.r1()),
      static_cast<float>(e.r2()),
      static_cast<float>(toDegree(e.o())),
      col, penWidth);
  }


} /* namespace DO */