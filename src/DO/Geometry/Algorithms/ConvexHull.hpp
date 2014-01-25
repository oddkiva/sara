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

#include <DO/Geometry/Objects/Polygon.hpp>

namespace DO {
  
  namespace internal {

    typedef std::pair<Point2d, double> PtCotg;
    
    void sortPointsByPolarAngle(Point2d *points, PtCotg *workArray,
                                int numPoints);

  } /* namespace internal */

  std::vector<Point2d> grahamScanConvexHull(const std::vector<Point2d>& points);

} /* namespace DO */
