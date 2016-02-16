// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Geometry.hpp>


namespace DO { namespace Sara {

  void draw_line_segment(const LineSegment& s, const Color3ub& c, int pen_width)
  {
    draw_line(s.p1(), s.p2(), c, pen_width);
  }

  void draw_bbox(const BBox& b, const Color3ub& color, int pen_width)
  {
    Point2i tl(b.top_left().cast<int>());
    Point2i br(b.bottom_right().cast<int>());
    Point2i sz(br-tl);
    draw_rect(tl.x(), tl.y(), sz.x(), sz.y(), color, pen_width);
  }

  void draw_poly(const std::vector<Point2d>& p, const Color3ub& color,
                int pen_width)
  {
    for (size_t i1 = 0, i2 = p.size()-1; i1 != p.size(); i2=i1++)
      draw_line(p[i1], p[i2], color, pen_width);
  }

  void draw_ellipse(const Ellipse& e, const Color3ub& col, int pen_width)
  {
    // Ellipse...
    draw_ellipse(e.center(), e.radius1(), e.radius2(),
                to_degree(e.orientation()), col, pen_width);
    // Arrow...
    Vector2d u(unit_vector2(e.orientation()));
    u *= e.radius1()*1.1;
    Point2d a, b;
    a = e.center();
    b = e.center() + u;
    draw_arrow(a, b, col, pen_width);
    // Center...
    draw_circle(e.center(), 5., col, pen_width);
  }

  void draw_affine_cone(const AffineCone2& K, double arrow_length,
                        const Color3ub& color)
  {
    const Point2d& v = K.vertex();
    Point2d a, b;
    a = v + arrow_length*K.alpha();
    b = v + arrow_length*K.beta();
    draw_arrow(v, a, color);
    draw_arrow(v, b, color);
  }


} /* namespace Sara */
} /* namespace DO */
