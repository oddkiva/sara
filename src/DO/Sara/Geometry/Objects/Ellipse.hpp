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

#ifndef DO_SARA_GEOMETRY_ELLIPSE_HPP
#define DO_SARA_GEOMETRY_ELLIPSE_HPP

#define _USE_MATH_DEFINES

#include <cmath>

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Geometry/Objects/Quad.hpp>


namespace DO { namespace Sara {

  //! Ellipse class
  class Ellipse
  {
  public:
    //! Default constructor.
    Ellipse()
    {
    }

    //! Constructor.
    Ellipse(double radius1, double radius2, double orientation,
            const Point2d& center)
      : a_(radius1), b_(radius2), o_(orientation), c_(center) {}

    double radius1() const { return a_; }
    double radius2() const { return b_; }
    double orientation() const { return o_; }
    const Point2d& center() const { return c_; }

    double& radius1() { return a_; }
    double& radius2() { return b_; }
    double& orientation() { return o_; }
    Point2d& center() { return c_; }

    //! Get the radial vector at angle $\theta$ w.r.t. orientation $o$ of ellipse.
    Vector2d rho(double theta) const;

    //! Get point on ellipse at angle $\theta$ w.r.t. orientation $o$ of ellipse.
    Point2d operator()(double theta) const;

    /*!
      Retrieve relative orientation of point $p$ w.r.t. orientation
      $o$ of ellipse.
     */
    friend double orientation(const Point2d& p, const Ellipse& e);

    //! Polar antiderivative.
    friend inline double polar_antiderivative(const Ellipse& e, double theta)
    {
      const double y = (e.b_-e.a_)*sin(2*theta);
      const double x = (e.b_+e.a_) + (e.b_-e.a_)*cos(2*theta);
      return e.a_*e.b_*0.5*( theta - atan2(y,x) );
    }

    /*!
      This function should be used instead to compute the **positive** area
      of an ellipse sector which we define as the region bounded by:
      - the **counter-clockwise** oriented arc going **from** the endpoint
        $M(\theta0)$ **to** endpoint $M(\theta_1)$.
      - line segments connecting the center of the ellipse and the endpoints
        of the arc.

      $\theta_0$ and $\theta_1$ are required to be in the range $]\pi, \pi]$ but
      it does not matter if $\theta_0 > \theta_1$.
     */
    friend double sector_area(const Ellipse& e, double theta0, double theta1)
    {
      return polar_antiderivative(e, theta1) - polar_antiderivative(e, theta0);
    }

    /*!
      An elliptic segment is a region bounded by an arc and the chord connecting
      the arc's endpoints.
     */
    friend double segment_area(const Ellipse& e, double theta0, double theta1);

    //! Ellipse area.
    friend inline double area(const Ellipse& e)
    {
      return M_PI*e.a_*e.b_;
    }

    //! Shape matrix.
    friend inline Matrix2d shape_matrix(const Ellipse& e)
    {
      const Eigen::Rotation2D<double> R(e.o_);
      Vector2d D( 1./(e.a_*e.a_), 1./(e.b_*e.b_) );
      return R.matrix()*D.asDiagonal()*R.matrix().transpose();
    }

    //! Checks if point is inside ellipse.
    friend inline bool inside(const Point2d& p, const Ellipse& e)
    {
      return (p-e.c_).transpose()*shape_matrix(e)*(p-e.c_) < 1.;
    }

    //! Compute rotated bbox of the ellispe.
    friend Quad oriented_bbox(const Ellipse& e);

    //! I/O.
    friend std::ostream& operator<<(std::ostream& os, const Ellipse& e);

  private:
    double a_, b_;
    double o_;
    Point2d c_;
  };

  //! Compute the ellipse from the conic equation
  Ellipse construct_from_shape_matrix(const Matrix2d& shapeMat,
                                      const Point2d& c);

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_GEOMETRY_ELLIPSE_HPP */
