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

#ifndef DO_SARA_GEOMETRY_ALGORITHMS_CONVEXHULL_HPP
#define DO_SARA_GEOMETRY_ALGORITHMS_CONVEXHULL_HPP

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Geometry/Objects/Polygon.hpp>


namespace DO { namespace Sara { namespace Detail {

    using PtCotg = std::pair<Point2d, double>;

    void sort_points_by_polar_angle(Point2d *points, PtCotg *workArray,
                                    std::size_t num_points);

} /* namespace Detail */
} /* namespace Sara */
} /* namespace DO */


namespace DO { namespace Sara {

  DO_SARA_EXPORT
  std::vector<Point2d>
  graham_scan_convex_hull(const std::vector<Point2d>& points);

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_GEOMETRY_ALGORITHMS_CONVEXHULL_HPP */
