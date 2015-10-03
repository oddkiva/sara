// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_SARA_GEOMETRY_OBJECTS_SPHERE_HPP
#define DO_SARA_GEOMETRY_OBJECTS_SPHERE_HPP

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  class Sphere
  {
    Point3d c_;
    double r_;
  public:
    Sphere(const Point3d& c, double r) : c_(c), r_(r) {}
    const Point3d& center() const { return c_; }
    double radius() const { return r_; }

    friend bool inside(const Point3d& x, const Sphere& S)
    { return (x - S.c_).squaredNorm() < S.radius()*S.radius(); }
  };

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_GEOMETRY_OBJECTS_SPHERE_HPP */
