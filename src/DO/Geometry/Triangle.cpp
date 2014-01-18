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

namespace DO {

  Triangle::Triangle(const Point2d& a, const Point2d& b, const Point2d& c)
  {
    v[0] = a; v[1] = b; v[2] = c;
    Matrix2d U;
    U.col(0) = b-a;
    U.col(1) = c-a;
    if (U.determinant() < 0)
      std::swap(v[1], v[2]);

    for (int i = 0; i < 3; ++i)
    {
      n[i] = v[(i+1)%3] - v[i];
      std::swap(n[i].x(), n[i].y());
      n[i].y() = -n[i].y();
    }
  }

  double Triangle::area() const
  {
    Matrix2d M;
    M.col(0) = v[1]-v[0];
    M.col(1) = v[2]-v[0];
    return 0.5*abs(M.determinant());
  }

  bool isInside(const Point2d& p, const Triangle& t)
  {
    for (int i = 0; i < 3; ++i)
    {
      Vector2d u(p-v[i]);
      if (n[i].dot(u) > 1e-10)
        return false;
    }
    return true;
  }

  void drawTriangle(const Rgb8& col)
  {
    drawLine(v[0], v[1], col, penWidth);
    drawLine(v[1], v[2], col, penWidth);
    drawLine(v[0], v[2], col, penWidth);
  }

}