// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

// 2D objects but their generalization to higher dimensions is straightforward.
// \todo: see if generalization can be implemented easily by implementing a Kernel.
#include <DO/Sara/Geometry/Objects/LineSegment.hpp>
#include <DO/Sara/Geometry/Objects/BBox.hpp>
#include <DO/Sara/Geometry/Objects/Polygon.hpp>
#include <DO/Sara/Geometry/Objects/Triangle.hpp>
#include <DO/Sara/Geometry/Objects/Quad.hpp>
#include <DO/Sara/Geometry/Objects/Ellipse.hpp>
#include <DO/Sara/Geometry/Objects/Cone.hpp>

// 3D objects
#include <DO/Sara/Geometry/Objects/Cube.hpp>
#include <DO/Sara/Geometry/Objects/HalfSpace.hpp>
#include <DO/Sara/Geometry/Objects/Sphere.hpp>

// Constructive Solid Geometry.
#include <DO/Sara/Geometry/Objects/CSG.hpp>
