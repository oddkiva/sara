// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#define _USE_MATH_DEFINES

#include <cmath>
#include <vector>

#include <DO/Sara/Geometry/Objects/Ellipse.hpp>


namespace DO { namespace Sara {

  /*!
   * Compute the approximate intersection-union ratio by approximating ellipses
   * with polygons.
   */
  std::vector<Point2d> approximage_intersection(const Ellipse& e1,
                                                const Ellipse& e2,
                                                int ellipse_discretization);

  /*!
   * Compute the approximate Jaccard distance by approximating ellipses with
   * polygons.
   */
  double approximate_jaccard_similarity(const Ellipse& e1,
                                        const Ellipse& e2,
                                        int ellipse_discretization = 36,
                                        double limit = 1e9);

  //! Compute intersection points between two ellipses and return the number of
  //! intersection points.
  int compute_intersection_points(Point2d intersections[],
                                  const Ellipse& e1,
                                  const Ellipse& e2);

  /*!
    Compute the intersection union ratio exactly.
    CAUTION: Numerical issues are not totally solved. We are almost there...
    Investigation is still ongoing.
   */
  double analytic_intersection(const Ellipse& e1, const Ellipse& e2);

  /*!
    Compute the intersection union ratio exactly.
    CAUTION: Numerical issues are not totally solved. We are almost there...
    Investigation is still ongoing.
   */
  double analytic_jaccard_similarity(const Ellipse& e1, const Ellipse& e2);


} /* namespace Sara */
} /* namespace DO */
