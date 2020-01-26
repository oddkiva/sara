// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
/*!
 *  This file implements a part of the method published in:
 *
 *  Efficient and Scalable 4th-order Match Propagation
 *  David Ok, Renaud Marlet, and Jean-Yves Audibert.
 *  ACCV 2012, Daejeon, South Korea.
 */

#pragma once

#include <DO/Sara/Features.hpp>
#include <DO/Sara/Match.hpp>
#include <DO/Sara/Geometry.hpp>


namespace DO::Sara {

  void get_triangle_angles_in_degree(double angles[3], const Point2d t[3]);

  // Convenience function.
  Matrix3f affinity_from_x_to_y(const Match& a, const Match& b, const Match& c);

  // Remember Jacobian $\mathbf{J} = (\nabla \mathbf{H})^T$.
  Matrix2d jacobian(const Matrix3d& H, const Vector2d& x);

  // It looks something is wrong here. Investigate...
  Matrix3d affinity(const Matrix3d& H, const Vector2d& x);

  // Helper function.
  Ellipse ellipse_from_oeregion(const OERegion& f);

  OERegion transform_oeregion(const OERegion& f, const Matrix3f& H);

  double compute_orientation(const Match& m, const Matrix3f& H);

  double angle_difference_in_radian(const OERegion& H_x, const OERegion& y);

  float center_difference(const OERegion& H_x, const OERegion& y);

  // Actually more reliable but so much slower, also the times when the
  // analytical computation of ellipses intersection is wrong (because of
  // numerical issues) are VERY seldom (less than 1%).
  void compare_oeregions(float& dist, double& diff_angle_radian,
                         double& overlapRatio, const OERegion& H_x,
                         const OERegion& y, bool approxEllInterArea = false);

}  // namespace DO::Sara
