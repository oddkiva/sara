// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Match.hpp>
#include <DO/Sara/Graphics.hpp>


namespace DO { namespace Sara {

  void PairWiseDrawer::displayImages() const
  {
    using namespace std;
    display(image1, (offF(0)*scale(0)).cast<int>(), z1);
    display(image2, (offF(1)*scale(0)).cast<int>(), z2);
  }

  void PairWiseDrawer::drawPoint(int i, const Point2f& p, const Color3ub& c, int r) const
  {
    assert(i == 0 || i == 1);
    fillCircle( (p+offF(i))*scale(i), float(r), c);
  }

  void PairWiseDrawer::drawLine(int i, const Point2f& pa, const Point2f& pb,
                                const Color3ub& c, int penWidth) const
  {
    assert(i == 0 || i == 1);
    Vector2f a( (pa+offF(i))*scale(i) );
    Vector2f b( (pb+offF(i))*scale(i) );
    DO::drawLine(a, b, c, penWidth);
  }

  void PairWiseDrawer::drawArrow(int i, const Point2f& pa, const Point2f& pb, const Color3ub& c, int penWidth) const
  {
    assert(i == 0 || i == 1);
    Vector2f a( (pa+offF(i))*scale(i) );
    Vector2f b( (pb+offF(i))*scale(i) );
    DO::drawLine(a, b, c, penWidth);
  }

  void PairWiseDrawer::drawTriangle(int i, const Point2f& pa, const Point2f& pb, const Point2f& pc,
                                    const Color3ub& c, int r) const
  {
    assert(i == 0 || i ==1);
    drawLine(i, pa, pb, c, r);
    drawLine(i, pb, pc, c, r);
    drawLine(i, pa, pc, c, r);
  }

  void PairWiseDrawer::drawRect(int i, const Point2f& p1, const Point2f& p2, int r, const Color3ub& c) const
  {
    assert(i==0 || i==1);
    Point2f s, e, d;
    s = scale(i)*p1+offF(i)*scale(0);
    e = scale(i)*p2+offF(i)*scale(0);
    d = e-s;

    DO::drawRect(s.x(), s.y(), d.x(), d.y(), c, r);
  }

  void PairWiseDrawer::drawLineFromEqn(int i, const Vector3f& eqn, const Color3ub& c, int r) const
  {
    assert(i == 0 || i ==1);
    Point2f a, b;
    a.x() = 0;
    a.y() = -eqn[2]/eqn[1];

    b.x() = (i==0) ? float(image1.width()) : float(image2.width());
    b.y() = -( eqn[0]*b.x() + eqn[2] ) / eqn[1];

    drawLine(i, a, b, c, r);
  }

  void PairWiseDrawer::drawFeature(int i, const OERegion& f, const Color3ub& c) const
  {
    assert(i == 0 || i == 1);
    f.draw(c, z1, offF(i));
  }

  void PairWiseDrawer::drawMatch(const Match& m, const Color3ub& c, bool drawLine) const
  {
    drawFeature(0, m.x(), c);
    drawFeature(1, m.y(), c);

    if(drawLine)
    {
      Vector2f a,b;
      a = scale(0)*m.posX(); b = scale(1)*(m.posY()+offF(1));
      DO::drawLine(a, b, c);
    }
  }

} /* namespace Sara */
} /* namespace DO */
