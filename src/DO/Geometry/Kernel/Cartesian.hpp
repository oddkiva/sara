// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Core/EigenExtension.hpp>


//! \todo: Implement simple geometric kernels.
namespace DO { namespace Concept {

  template <typename FT_, int N>
  struct Cartesian
  {
    typedef FT_ FieldType;
    typedef Matrix<FieldType, N, 1> Point, Vector;
  };

  template <typename FT_, int N>
  struct Homogeneous
  {
    typedef FT_ FieldType;
    typedef Matrix<FieldType, N+1, 1> Point, Vector;
  };

  // ======================================================================== //
  // Objects
  //
  // 2D and 3D
  template <typename Kernel> class BBox;
  template <typename Kernel> class Line;
  template <typename Kernel> class Quad;
  template <typename Kernel> class Triangle;
  template <typename Kernel> class LineSegment;
  template <typename Kernel> class Circle;
  template <typename Kernel> class Direction;

  // Only 3D
  template <typename Kernel> class Tetrahedron;
  template <typename Kernel> class Cube;
  template <typename Kernel> class Ray;
  template <typename Kernel> class Sphere;

  // In any dimension
  template <typename Kernel> class Polytope; // (polygon 2D, polyhedron 3D)
  template <typename Kernel> class Cone;
  template <typename Kernel> class AffineCone;
  template <typename Kernel> class HyperPlane; // (line 2D, plane 2D)
  template <typename Kernel> class HyperSphere; // (Disc 2D, Ball
  template <typename Kernel> class OpenBall;
  template <typename Kernel> class Metric;


  // ======================================================================== //
  // Objects
  //
  template <typename Kernel>
  bool intersection(const Line<Kernel>& line1, const Line<Kernel>& line2,
                    typename Kernel::Point& point);

  template <typename Kernel>
  double area(const Triangle<Kernel>& triangle);

} /* namespace Concept */
} /* namespace DO */