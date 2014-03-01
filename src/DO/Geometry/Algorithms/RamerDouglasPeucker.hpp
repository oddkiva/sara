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

#ifndef DO_GEOMETRY_ALGORITHMS_RAMERDOUGLASPEUCKER_HPP
#define DO_GEOMETRY_ALGORITHMS_RAMERDOUGLASPEUCKER_HPP

#include <DO/Core/EigenExtension.hpp>
#include <DO/Core/StdVectorHelpers.hpp>

namespace DO {

  std::vector<Point2d>
  ramerDouglasPeucker(const std::vector<Point2d>& contours, double eps);

}

#endif /* DO_GEOMETRY_ALGORITHMS_RAMERDOUGLASPEUCKER_HPP */