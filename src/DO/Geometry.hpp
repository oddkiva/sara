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

#ifndef DO_GEOMETRY_HPP
#define DO_GEOMETRY_HPP

//! Some useful mathematical tools.
#include "Geometry/Tools/Projective.hpp"
#include "Geometry/Tools/Utilities.hpp"
#include "Geometry/Tools/Metric.hpp"
#include "Geometry/Tools/Polynomial.hpp"
#include "Geometry/Tools/PolynomialRoots.hpp"

//! Basic data structures for computational geometry.
#include "Geometry/LineSegment.hpp"
#include "Geometry/BBox.hpp"
#include "Geometry/Polygon.hpp"
#include "Geometry/Triangle.hpp"
#include "Geometry/Quad.hpp"
#include "Geometry/Ellipse.hpp"
//! Basic computational geometry algorithms
#include "Geometry/ConvexHull.hpp"
#include "Geometry/EllipseIntersection.hpp"
#include "Geometry/SutherlandHodgman.hpp"

//! Graphics.
#include "Geometry/Graphics/DrawPolygon.hpp"

#endif /* DO_GEOMETRY_HPP */