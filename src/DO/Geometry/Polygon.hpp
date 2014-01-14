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

#include <DO/Core.hpp>
#include <vector>

namespace DO {

  std::vector<Point2d> grahamScanConvexHull(const std::vector<Point2d>& points);

  double area(const std::vector<Point2d>& ccwPolygon);

} /* namespace DO */