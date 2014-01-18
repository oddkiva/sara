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
#include <DO/Graphics.hpp>

using namespace std;

namespace DO {

  std::ostream& operator<<(std::ostream& os, const BBox& bbox)
  {
    os << "top-left " << bbox.tl_.transpose() << endl;
    os << "bottom-right " << bbox.br_.transpose() << endl;
    return os;
  }
  
  bool inside(const Point2d& p, const BBox& bbox)
  {
    return 
      p.x() >= bbox.topLeft().x() && p.x() <= bbox.bottomRight().x() &&
      p.y() >= bbox.topLeft().y() && p.y() <= bbox.bottomRight().y() ;
  }

  bool degenerate(const BBox& bbox, double areaThres)
  {
    return (bbox.area() < areaThres);
  }
  
  void drawBBox(const BBox& b, const Color3ub& color, double z, int penWidth)
  {
    Point2d tl(z*b.topLeft());
    Point2d br(z*b.bottomRight());
    Point2d sz(br-tl);
    drawRect(tl.cast<int>()(0), tl.cast<int>()(1),
             sz.cast<int>()(0), sz.cast<int>()(1),
             color, penWidth);
  }

} /* namespace DO */