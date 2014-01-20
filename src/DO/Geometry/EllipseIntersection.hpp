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

#ifndef DO_GEOMETRY_ELLIPSEINTERSECTION_HPP
#define DO_GEOMETRY_ELLIPSEINTERSECTION_HPP

#include <DO/Geometry/Ellipse.hpp>
#define _USE_MATH_DEFINES
#include <cmath>

namespace DO {

  //! Compute the intersection union ratio approximately
  double approximateIntersectionUnionRatio(const Ellipse& e1, const Ellipse& e2,
                                           int n = 36,
                                           double limit = 1e9);

  //! Check polynomial solvers.
  void getEllipseIntersections(Point2d intersections[4], int& numInter,
                               const Ellipse& e1, const Ellipse& e2);

  double convexSectorArea(const Ellipse& e, const Point2d pts[]);

  double analyticInterUnionRatio(const Ellipse& e1, const Ellipse& e2);


} /* namespace DO */

#endif /* DO_GEOMETRY_ELLIPSEINTERSECTION_HPP */