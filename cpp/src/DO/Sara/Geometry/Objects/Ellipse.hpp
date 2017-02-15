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

//! @file

#pragma once

#include <cmath>

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/EigenExtension.hpp>

#include <DO/Sara/Geometry/Objects/Quad.hpp>


namespace DO { namespace Sara {

  //! Ellipse class
  class DO_SARA_EXPORT Ellipse
  {
  public:
    //! @brief Default constructor.
    Ellipse() = default;

    //! @brief Constructor.
    Ellipse(double radius1, double radius2, double orientation,
            const Point2d& center)
      : _a{ radius1 }
      , _b{ radius2 }
      , _o{ orientation }
      , _c{ center }
    {
    }

    //! @{
    //! @brief Data member accessor.
    double radius1() const { return _a; }
    double radius2() const { return _b; }
    double orientation() const { return _o; }
    const Point2d& center() const { return _c; }

    double& radius1() { return _a; }
    double& radius2() { return _b; }
    double& orientation() { return _o; }
    Point2d& center() { return _c; }
    //! @}

    //! @brief Get the radial vector at angle $\theta$ w.r.t. orientation $o$
    //! of ellipse.
    Vector2d rho(double theta) const;

    //! @brief Get point on ellipse at angle $\theta$ w.r.t. orientation $o$ of
    //! ellipse.
    Point2d operator()(double theta) const;

    /*!
      @brief Retrieve relative orientation of point $p$ w.r.t. orientation
      $o$ of ellipse.
     */
    DO_SARA_EXPORT
    friend double orientation(const Point2d& p, const Ellipse& e);

    //! @brief Polar anti derivative.
    friend inline double polar_antiderivative(const Ellipse& e, double theta)
    {
      const double y = (e._b-e._a)*sin(2*theta);
      const double x = (e._b+e._a) + (e._b-e._a)*cos(2*theta);
      return e._a*e._b*0.5*( theta - atan2(y,x) );
    }

    /*!
      This function should be used instead to compute the **positive** area
      of an ellipse sector which we define as the region bounded by:
      - the **counter-clockwise** oriented arc going **from** the endpoint
        $M(\theta0)$ **to** endpoint $M(\thet_a1)$.
      - line segments connecting the center of the ellipse and the endpoints
        of the arc.

      $\thet_a0$ and $\thet_a1$ are required to be in the range $]\pi, \pi]$ but
      it does not matter if $\thet_a0 > \thet_a1$.
     */
    friend double sector_area(const Ellipse& e, double theta0, double theta1)
    {
      return polar_antiderivative(e, theta1) - polar_antiderivative(e, theta0);
    }

    /*!
      An elliptic segment is a region bounded by an arc and the chord connecting
      the arc's endpoints.
     */
    DO_SARA_EXPORT
    friend double segment_area(const Ellipse& e, double theta0, double theta1);

    //! @brief Returns the ellipse area.
    friend inline double area(const Ellipse& e)
    {
      return M_PI*e._a*e._b;
    }

    //! @brief Returns the shape matrix of the ellipse.
    friend inline Matrix2d shape_matrix(const Ellipse& e)
    {
      const Eigen::Rotation2D<double> R(e._o);
      const Vector2d D{ 1. / (e._a*e._a), 1. / (e._b*e._b) };
      return R.matrix()*D.asDiagonal()*R.matrix().transpose();
    }

    //! @brief Check whether the point is inside ellipse.
    inline bool contains(const Point2d& p) const
    {
      return (p - _c).transpose()*shape_matrix(*this)*(p - _c) < 1.;
    }

    //! @brief Computes the rotated BBox of the ellipse.
    DO_SARA_EXPORT
    friend Quad oriented_bbox(const Ellipse& e);

    //! @brief I/O.
    DO_SARA_EXPORT
    friend std::ostream& operator<<(std::ostream& os, const Ellipse& e);

  private:
    double _a, _b;
    double _o;
    Point2d _c;
  };

  //! Compute the ellipse from the conic equation
  DO_SARA_EXPORT
  Ellipse construct_from_shape_matrix(const Matrix2d& shape_matrix,
                                      const Point2d& c);

} /* namespace Sara */
} /* namespace DO */
