// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Match.hpp>
#include <DO/Sara/Graphics.hpp>


namespace DO { namespace Sara {

  void PairWiseDrawer::display_images() const
  {
    display(image1, (offF(0) * scale(0)).cast<int>(), _z1);
    display(image2, (offF(1) * scale(0)).cast<int>(), _z2);
  }

  void PairWiseDrawer::draw_point(int i, const Point2f& p, const Color3ub& c, int r) const
  {
    assert(i == 0 || i == 1);
    fill_circle((p + offF(i)) * scale(i), float(r), c);
  }

  void PairWiseDrawer::draw_line(int i, const Point2f& pa, const Point2f& pb,
                                const Color3ub& c, int penWidth) const
  {
    assert(i == 0 || i == 1);
    const Vector2f a = (pa + offF(i)) * scale(i);
    const Vector2f b = (pb + offF(i)) * scale(i);
    Sara::draw_line(a, b, c, penWidth);
  }

  void PairWiseDrawer::draw_arrow(int i, const Point2f& pa, const Point2f& pb, const Color3ub& c, int penWidth) const
  {
    assert(i == 0 || i == 1);
    const Vector2f a = (pa + offF(i)) * scale(i);
    const Vector2f b = (pb + offF(i)) * scale(i);
    Sara::draw_arrow(a, b, c, penWidth);
  }

  void PairWiseDrawer::draw_triangle(int i, const Point2f& pa, const Point2f& pb, const Point2f& pc,
                                    const Color3ub& c, int r) const
  {
    assert(i == 0 || i ==1);
    draw_line(i, pa, pb, c, r);
    draw_line(i, pb, pc, c, r);
    draw_line(i, pa, pc, c, r);
  }

  void PairWiseDrawer::draw_rect(int i, const Point2f& p1, const Point2f& p2, int r, const Color3ub& c) const
  {
    assert(i==0 || i==1);
    const Point2i s = (scale(i) * p1 + offF(i) * scale(0))
                          .array()
                          .round()
                          .matrix()
                          .cast<int>();
    const Point2i e = (scale(i) * p2 + offF(i) * scale(0))
                          .array()
                          .round()
                          .matrix()
                          .cast<int>();
    const Point2i d = e - s;

    Sara::draw_rect(s.x(), s.y(), d.x(), d.y(), c, r);
  }

  void PairWiseDrawer::draw_line_from_eqn(int i, const Vector3f& eqn, const Color3ub& c, int r) const
  {
    assert(i == 0 || i ==1);
    Point2f a, b;
    a.x() = 0;
    a.y() = -eqn[2] / eqn[1];

    b.x() = (i == 0) ? float(image1.width()) : float(image2.width());
    b.y() = -(eqn[0] * b.x() + eqn[2]) / eqn[1];

    draw_line(i, a, b, c, r);
  }

  void PairWiseDrawer::draw_feature(int i, const OERegion& f, const Color3ub& c) const
  {
    assert(i == 0 || i == 1);
    draw(f, c, _z1, offF(i));
  }

  void PairWiseDrawer::draw_match(const Match& m, const Color3ub& c, bool drawLine) const
  {
    draw_feature(0, m.x(), c);
    draw_feature(1, m.y(), c);

    if(drawLine)
    {
      Vector2f a,b;
      a = scale(0) * m.x_pos();
      b = scale(1) * (m.y_pos() + offF(1));
      Sara::draw_line(a, b, c);
    }
  }

} /* namespace Sara */
} /* namespace DO */
