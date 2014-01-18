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

#ifndef DO_GEOMETRY_TRIANGLE_HPP
#define DO_GEOMETRY_TRIANGLE_HPP

#include <DO/Core/EigenExtension.hpp>
#include <DO/Core/Color.hpp>

namespace DO {

  // Triangle (a,b,c) enumerated in CCW order.
  class Triangle : public SmallPolygon<3>
  {
  public:
    Triangle(const Point2d& a, const Point2d& b, const Point2d& c);
  };
  
  bool inside(const Point2d& p, const Triangle& t);
  
  void drawTriangle(const Triangle& t, const Rgb8& col = Red8,
                    int penWidth = 1);

} /* namespace DO */

#endif /* DO_TRIANGLE_HPP */