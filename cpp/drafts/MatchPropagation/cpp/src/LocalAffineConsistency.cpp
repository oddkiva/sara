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

#include "LocalAffineConsistency.hpp"

#include <DO/Sara/Geometry.hpp>


using namespace std;


namespace DO::Sara {

  void get_triangle_angles_in_degree(double angles[3], const Point2d t[3])
  {
    for (int i = 0; i < 3; ++i)
    {
      Vector2d u(t[(i + 1) % 3] - t[i]), v(t[(i + 2) % 3] - t[i]);
      u.normalize();
      v.normalize();
      angles[i] = to_degree(acos(u.dot(v)));
    }
    sort(angles, angles + 3);
  }

  Matrix3f affinity_from_x_to_y(const Match& a, const Match& b, const Match& c)
  {
    return affine_transform_2(a.x_pos(), b.x_pos(), c.x_pos(),  //
                              a.y_pos(), b.y_pos(), c.y_pos());
  }

  // Remember Jacobian $\mathbf{J} = (\nabla \mathbf{H})^T$.
  Matrix2d jacobian(const Matrix3d& H, const Vector2d& x)
  {
    Matrix2d J;
    Vector3d xh;
    xh << x, 1.;
    double h1_xh = H.row(0) * xh;
    double h2_xh = H.row(1) * xh;
    double h3_xh = H.row(2) * xh;

    RowVector2d h1_t = H.row(0).head(2);
    RowVector2d h2_t = H.row(1).head(2);
    RowVector2d h3_t = H.row(2).head(2);

    J.row(0) = (h1_t * h3_xh - h1_xh * h3_t) / (h3_xh * h3_xh);
    J.row(1) = (h2_t * h3_xh - h2_xh * h3_t) / (h3_xh * h3_xh);

    return J;
  }

  // It looks something is wrong here. Investigate...
  Matrix3d affinity(const Matrix3d& H, const Vector2d& x)
  {
    Vector3d H_xh;
    H_xh << x(0), x(1), 1.;
    H_xh = H * H_xh;
    H_xh /= H_xh(2);

    Vector2d Hx;
    Hx = H_xh.head(2);
    Matrix2d Jx;
    Jx = jacobian(H, x);

    Matrix3d A;
    A.setZero();
    A.block(0, 0, 2, 2) = Jx;
    A.block(0, 2, 2, 1) = Hx - Jx * x;
    A(2, 2) = 1.;
    return A;
  }

  // Helper function.
  Ellipse ellipse_from_oeregion(const OERegion& f)
  {
    return construct_from_shape_matrix(f.shape_matrix.cast<double>(),
                                       f.center().cast<double>());
  }

  OERegion transform_oeregion(const OERegion& f, const Matrix3f& H)
  {
    const Vector2f& x = f.center();
    const Matrix2f& Sigma = f.shape_matrix;
    float o = f.orientation;

    Vector3f H_xh;
    H_xh << x, 1;
    H_xh = H * H_xh;
    H_xh /= H_xh(2);
    Vector2f H_x;
    H_x = H_xh.head(2);

    Matrix2f L, invL;
    L = jacobian(H.cast<double>(), x.cast<double>()).cast<float>();
    invL = L.inverse();
    Vector2f L_ox(L * unit_vector2(o));
    L_ox.normalize();

    OERegion phi_f;
    phi_f.center() = H_xh.head(2);
    phi_f.shape_matrix = invL.transpose() * Sigma * invL;
    phi_f.orientation = atan2(L_ox(1), L_ox(0));

    return phi_f;
  }

  double compute_orientation(const Match& m, const Matrix3f& Hf)
  {
    double anglex = m.x().orientation;
    Vector2d ox(cos(anglex), sin(anglex));

    Matrix2d L(jacobian(Hf.cast<double>(), m.x_pos().cast<double>()));
    Vector2d L_ox(L * ox);
    L_ox.normalize();

    anglex = atan2(L_ox(1), L_ox(0));

    return anglex;
  }

  double angle_difference_in_radian(const OERegion& H_x, const OERegion& y)
  {
    Vector2d o_H_x(unit_vector2(double(H_x.orientation)));
    Vector2d o_y(unit_vector2(double(y.orientation)));
    double diff_angle = acos(o_y.dot(o_H_x));
    return diff_angle;
  }

  float center_difference(const OERegion& H_x, const OERegion& y)
  {
    return (H_x.center() - y.center()).norm();
  }

  void compare_oeregions(float& dist, double& diff_angle_radian,
                         double& overlap_ratio, const OERegion& H_x,
                         const OERegion& y, bool approximate_ellipse_intersection)
  {
    Ellipse H_Sx = ellipse_from_oeregion(H_x);
    Ellipse Sy = ellipse_from_oeregion(y);

    dist = center_difference(H_x, y);
    overlap_ratio = approximate_ellipse_intersection
                       ? approximate_jaccard_similarity(H_Sx, Sy)
                       : analytic_jaccard_similarity(H_Sx, Sy);
    if (overlap_ratio < 0.)
      overlap_ratio = 0.;
    if (overlap_ratio > 1.)
      overlap_ratio = 1.;
    diff_angle_radian = angle_difference_in_radian(H_x, y);
  }

}  // namespace DO::Sara
