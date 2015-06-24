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

#ifndef DO_SARA_GEOMETRY_TRIANGLE_HPP
#define DO_SARA_GEOMETRY_TRIANGLE_HPP

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Geometry/Objects/Polygon.hpp>


namespace DO { namespace Sara {

  // Triangle (a,b,c) enumerated in CCW order.
  class DO_EXPORT Triangle : public SmallPolygon<3>
  {
  public:
    Triangle() : SmallPolygon<3>() {}
    Triangle(const Point2d& a, const Point2d& b, const Point2d& c);
  };

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_TRIANGLE_HPP */