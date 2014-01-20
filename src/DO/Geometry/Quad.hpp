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

#ifndef DO_GEOMETRY_QUAD_HPP
#define DO_GEOMETRY_QUAD_HPP

#include <DO/Core/EigenExtension.hpp>
#include <DO/Geometry/Polygon.hpp>
#include <DO/Geometry/BBox.hpp>

namespace DO {

  class Quad : public SmallPolygon<4>
  {
    typedef SmallPolygon<4> Base;
  public:
    Quad(const BBox& bbox);
    Quad(const Point2d& a, const Point2d& b,
         const Point2d& c, const Point2d& d);
  };

} /* namespace DO */

#endif /* DO_GEOMETRY_QUAD_HPP */