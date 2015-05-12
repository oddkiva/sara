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

#ifndef DO_GEOMETRY_OBJECTS_HPP
#define DO_GEOMETRY_OBJECTS_HPP

// 2D objects but their generalization to higher dimensions is straightforward.
// \todo: see if generalization can be implemented easily by implementing a Kernel.
#include <DO/Geometry/Objects/LineSegment.hpp>
#include <DO/Geometry/Objects/BBox.hpp>
#include <DO/Geometry/Objects/Polygon.hpp>
#include <DO/Geometry/Objects/Triangle.hpp>
#include <DO/Geometry/Objects/Quad.hpp>
#include <DO/Geometry/Objects/Ellipse.hpp>
#include <DO/Geometry/Objects/Cone.hpp>

// 3D objects
#include <DO/Geometry/Objects/Cube.hpp>
#include <DO/Geometry/Objects/HalfSpace.hpp>
#include <DO/Geometry/Objects/Sphere.hpp>

// Constructive Solid Geometry.
#include <DO/Geometry/Objects/CSG.hpp>


#endif /* DO_GEOMETRY_OBJECTS_HPP */