// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <cmath>
#include <vector>

#include <DO/Sara/Geometry/Objects/Ellipse.hpp>


namespace DO { namespace Sara {

  //! @addtogroup GeometryAlgorithms
  //! @{

  /*!
   * Compute the approximate intersection-union ratio by approximating ellipses
   * with polygons.
   */
  DO_SARA_EXPORT
  std::vector<Point2d> approximate_intersection(const Ellipse& e1,
                                                const Ellipse& e2,
                                                int ellipse_discretization);

  /*!
   *  Compute the approximate Jaccard distance by approximating ellipses with
   *  polygons.
   */
  DO_SARA_EXPORT
  double approximate_jaccard_similarity(const Ellipse& e1,
                                        const Ellipse& e2,
                                        int ellipse_discretization = 36,
                                        double limit = 1e9);

  //! Compute intersection points between two ellipses and return the number of
  //! intersection points.
  DO_SARA_EXPORT
  auto compute_intersection_points(const Ellipse& e1, const Ellipse& e2,
                                   bool polish_intersection_points = false)
      -> std::vector<Point2d>;

  /*!
     Compute the ellipse intersection area exactly.

     CAUTION: Numerical issues are not totally solved. We are almost there...
     @todo Investigate.

   */
  DO_SARA_EXPORT
  double analytic_intersection_area(const Ellipse& e1, const Ellipse& e2,
                                    bool polish_intersection_points = false);

  /*!
      Compute the intersection union ratio exactly.

      CAUTION: Numerical issues are not totally solved. We are almost there...
     @todo Investigate.
   */
  DO_SARA_EXPORT
  double analytic_jaccard_similarity(const Ellipse& e1, const Ellipse& e2,
                                     bool polish_intersection_points = false);

  //! @}

} /* namespace Sara */
} /* namespace DO */
