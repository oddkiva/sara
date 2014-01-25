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

#ifndef DO_GEOMETRY_ELLIPSE_HPP
#define DO_GEOMETRY_ELLIPSE_HPP

#define _USE_MATH_DEFINES
#include <cmath>

namespace DO {

  //! Ellipse class
  class Ellipse
  {
  public:
    Ellipse() {}
    Ellipse(double radius1, double radius2, double orientation,
            const Point2d& center)
      : a_(radius1), b_(radius2), o_(orientation), c_(center) {}

    double r1() const { return a_; }
    double r2() const { return b_; }
    double o() const { return o_; }
    const Point2d& c() const { return c_; }

    double& r1() { return a_; }
    double& r2() { return b_; }
    double& o() { return o_; }
    Point2d& c() { return c_; }

    //! Polar antiderivative.
    friend inline double polarAntiderivative(const Ellipse& e, double theta)
    {
      const double y = (e.b_-e.a_)*sin(2*theta);
      const double x = ((e.b_+e.a_)+(e.b_-e.a_)*cos(2*theta));
      return e.a_*e.b_*0.5*( theta - atan(y/x) );
    }
    //! Convex sector area: this function is deprecated.
    friend double convexSectorArea(const Ellipse& e, const Point2d pts[]);
    /*!
      This function should be used instead to compute the sector area of an ellipse
      going from the polar angle $\theta_1$ to $\theta_2$ w.r.t. to ellipse orientation
      $o$.
      $\theta_1$ and $\theta_2$ are required to be in the range $]-\pi, \pi]$ and it is
      required to have $\theta_1$ and $\theta_2$.

      Still the definition of the sector area remains ambiguous and we must specify the
      direction either *clockwise* == -1, or *counter-clockwise* == 1
     */
    friend double sectorArea(const Ellipse& e, double theta1, double theta2,
                             int direction);
    //! Ellipse area.
    friend inline double area(const Ellipse& e)
    { return M_PI*e.a_*e.b_; }
    //! Shape matrix.
    friend inline Matrix2d shapeMat(const Ellipse& e)
    {
      const Eigen::Rotation2D<double> R(e.o_);
      Vector2d D( 1./(e.a_*e.a_), 1./(e.b_*e.b_) );
      return R.matrix()*D.asDiagonal()*R.matrix().transpose();
    }
    friend inline bool inside(const Point2d& p, const Ellipse& e)
    { return (p-e.c_).transpose()*shapeMat(e)*(p-e.c_) < 1.; }
    //! Checks if the ellipse is not weird
    friend inline bool wellDefined(const Ellipse& e, double limit = 1e9)
    { return std::abs(e.a_) < limit && std::abs(e.b_) < limit; }
    //! I/O.
    friend std::ostream& operator<<(std::ostream& os, const Ellipse& e);

  private:
    double a_, b_;
    double o_;
    Point2d c_;
  };

  //! Compute the ellipse from the conic equation
  Ellipse fromShapeMat(const Matrix2d& shapeMat, const Point2d& c);

} /* namespace DO */

#endif /* DO_GEOMETRY_ELLIPSE_HPP */