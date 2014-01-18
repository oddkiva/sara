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

    double area() const { return 3.14159265358979323846*a_*b_; }

    // Polar antiderivative
    double F(double theta) const
    {
      return a_*b_*0.5*
           ( theta 
           - atan( (b_-a_)*sin(2*theta) / ((b_+a_)+(b_-a_)*cos(2*theta)) ) );
    }

    Matrix2d shapeMat() const;
    
    bool isWellDefined(double limit = 1e9) const
    { return (std::abs(r1()) < limit && std::abs(r2()) < limit); }

    void drawOnScreen(const Color3ub c, double scale = 1.) const;

  private:
    double a_, b_;
    double o_;
    Point2d c_;
  };
  
  
  inline bool isInside(const Point2d& p, const Ellipse& e) const
  {
    return (p-e.c()).transpose()*shapeMat()*(p-e.c()) < 1.;
  }

  //! I/O.
  std::ostream& operator<<(std::ostream& os, const Ellipse& e);

  //! Compute the ellipse from the conic equation
  Ellipse fromShapeMat(const Matrix2d& shapeMat, const Point2d& c);

  //! Compute the intersection union ratio approximately
  double approximateIntersectionUnionRatio(const Ellipse& e1, const Ellipse& e2,
                                           int n = 36,
                                           double limit = 1e9);

  //! Check polynomial solvers.
  //! TODO: Make a special library for polynomial solvers.
  void checkQuadraticEquationSolver();
  void checkCubicEquationSolver();
  void checkQuarticEquationSolver();

  void getEllipseIntersections(Point2d intersections[4], int& numInter,
                               const Ellipse& e1, const Ellipse& e2);

  double convexSectorArea(const Ellipse& e, const Point2d pts[]);

  double analyticInterUnionRatio(const Ellipse& e1, const Ellipse& e2);

  // Move somewhere instead.
  std::vector<Point2d> convexHull(const std::vector<Point2d>& points);
  double convexHullArea(const std::vector<Point2d>& points);


} /* namespace DO */

#endif /* DO_GEOMETRY_ELLIPSE_HPP */