// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <iostream>
#include <vector>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <DO/Sara/Geometry/Algorithms/EllipseIntersection.hpp>
#include <DO/Sara/Geometry/Algorithms/ConvexHull.hpp>
#include <DO/Sara/Geometry/Algorithms/SutherlandHodgman.hpp>
#include <DO/Sara/Geometry/Tools/PolynomialRoots.hpp>
#include <DO/Sara/Geometry/Tools/Utilities.hpp>
#include <DO/Sara/Geometry/Graphics/DrawPolygon.hpp>


using namespace std;


namespace DO { namespace Sara {

  // ======================================================================== //
  // Computation of intersecting points of intersecting ellipses
  // ======================================================================== //
  void print_conic_equation(double s[6])
  {
    cout << s[0] << " + " << s[1] << " x + " << s[2] << " y + ";
    cout << s[3] << " x^2 + " << s[4] << " xy + " << s[5] << " y^2 = 0" << endl;
  }

  void conic_equation(double s[6], const Ellipse & e)
  {
    const Matrix2d M(shape_matrix(e));
    s[0] = e.center().x()*M(0,0)*e.center().x()
         + 2*e.center().x()*M(0,1)*e.center().y()
         + e.center().y()*M(1,1)*e.center().y()
         - 1.0;
    s[1] =-2.0*(M(0,0)*e.center().x() + M(0,1)*e.center().y());
    s[2] =-2.0*(M(1,0)*e.center().x() + M(1,1)*e.center().y());
    s[3] = M(0,0);
    s[4] = 2.0*M(0,1);
    s[5] = M(1,1);
  }

  Polynomial<double, 4> quartic_equation(const double s[6], const double t[6])
  {
    double d[6][6];
    for(int i = 0; i < 6; ++i)
      for(int j = 0; j < 6; ++j)
        d[i][j] = s[i]*t[j] - s[j]*t[i];

    Polynomial<double, 4> u;
    u[0] = d[3][1]*d[1][0] - d[3][0]*d[3][0];
    u[1] = d[3][4]*d[1][0] + d[3][1]*(d[4][0]+d[1][2]) - 2*d[3][2]*d[3][0];
    u[2] = d[3][4]*(d[4][0]+d[1][2]) + d[3][1]*(d[4][2]-d[5][1]) - d[3][2]*d[3][2] - 2*d[3][5]*d[3][0];
    u[3] = d[3][4]*(d[4][2]-d[5][1]) + d[3][1]*d[4][5] - 2*d[3][5]*d[3][2];
    u[4] = d[3][4]*d[4][5] - d[3][5]*d[3][5];
    return u;
  }

  void sigma_polynomial(double sigma[3], const double s[6], double y)
  {
    sigma[0] = s[0]+s[2]*y+s[5]*y*y;
    sigma[1] = s[1]+s[4]*y;
    sigma[2] = s[3];
  }

  inline double compute_conic_expression(double x, double y, const double s[6])
  {
    return s[0] + s[1]*x + s[2]*y + s[3]*x*x + s[4]*x*y + s[5]*y*y;
  }

  // If imaginary part precision: 1e-6.
  // If the conic equation produces an almost zero value i.e.: 1e-3.
  // then we decide there is an intersection.
  pair<bool, Point2d> is_real_root(const complex<double>& y,
                                   const double s[6], const double t[6])
  {
    if ( abs(imag(y)) > 1e-2*abs(real(y)) )
      return make_pair(false, Point2d());
    const double realY = real(y);
    double sigma[3], tau[3];
    sigma_polynomial(sigma, s, realY);
    sigma_polynomial(tau, t, realY);
    const double x = (sigma[2]*tau[0] - sigma[0]*tau[2])
      / (sigma[1]*tau[2] - sigma[2]*tau[1]);

    if (fabs(compute_conic_expression(x, realY, s)) < 1e-2 &&
        fabs(compute_conic_expression(x, realY, t)) < 1e-2)
    {
      /*cout << "QO(x,y) = " << computeConicExpression(x, realY, s) << endl;
      cout << "Q1(x,y) = " << computeConicExpression(x, realY, t) << endl;*/
      return make_pair(true, Point2d(x,realY));
    }
    else
      return make_pair(false, Point2d());
  }

  int compute_intersection_points(Point2d intersections[4],
                                const Ellipse & e1, const Ellipse & e2)
  {
    // Rescale ellipse to try to improve numerical accuracy.
    Point2d center;
    center = 0.5*(e1.center() + e2.center());


    Ellipse ee1(e1), ee2(e2);
    ee1.center() -= center;
    ee2.center() -= center;

    double s[6], t[6];
    conic_equation(s, ee1);
    conic_equation(t, ee2);

    Polynomial<double, 4> u(quartic_equation(s, t));

    complex<double> y[4];
    roots(u, y[0], y[1], y[2], y[3]);


    int numInter = 0;
    for(int i = 0; i < 4; ++i)
    {
      pair<bool, Point2d> p(is_real_root(y[i], s, t));
      if(!p.first)
        continue;
      intersections[numInter] = p.second + center;
      ++numInter;
    }

    const double eps = 1e-2;
    const double squared_eps = eps*eps;
    auto identicalPoints = [&](const Point2d& p, const Point2d& q) {
      return (p-q).squaredNorm() < squared_eps;
    };

    auto it = unique(intersections, intersections+numInter, identicalPoints);
    return static_cast<int>(it - intersections);
  }

  void orientation(double *ori, const Point2d *pts, int numPoints,
                   const Ellipse& e)
  {
    const Vector2d u(unit_vector2(e.orientation()));
    const Vector2d v(-u(1), u(0));
    transform(pts, pts+numPoints, ori, [&](const Point2d& p) -> double
    {
      const Vector2d d(p-e.center());
      return atan2(v.dot(d), u.dot(d));
    });
  }


  // ======================================================================== //
  // Computation of the area of intersecting ellipses.
  // ======================================================================== //
  double analytic_intersection(const Ellipse& E_0, const Ellipse& E_1)
  {
    // Find the intersection points of the two ellipses.
    Point2d inter_pts[4];
#ifdef RESOLVED_NUMERICAL_ACCURACY
    int num_inter = compute_intersection_points(inter_pts, E_0, E_1);
#else
    /*if (numInter > 0)
    {
      CHECK(numInter);*/
      if (!active_window())
        set_antialiasing(create_window(512, 512));
      clear_window();
      draw_ellipse(E_0, Red8, 3);
      draw_ellipse(E_1, Blue8, 3);
      Quad Q_0(oriented_bbox(E_0));
      Quad Q_1(oriented_bbox(E_1));

      BBox b0(&Q_0[0], &Q_0[0]+4);
      BBox b1(&Q_1[0], &Q_1[0]+4);
      b0.top_left() = b0.top_left().cwiseMin(b1.top_left());
      b0.bottom_right() = b0.bottom_right().cwiseMax(b1.bottom_right());

      draw_quad(Q_0, Red8, 3);
      draw_quad(Q_1, Blue8, 3);
      draw_bbox(b0, Green8, 3);
      get_key();

      // now rescale the ellipse.
      Point2d center(b0.center());
      Vector2d delta(b0.sizes() - center);
      delta = delta.cwiseAbs();

      Ellipse EE_0, EE_1;
      Matrix2d S_0 = delta.asDiagonal()*shape_matrix(E_0)*delta.asDiagonal();
      Matrix2d S_1 = delta.asDiagonal()*shape_matrix(E_1)*delta.asDiagonal();

      CHECK(shape_matrix(E_0));
      CHECK(S_0);


      Vector2d c_0 = E_0.center() - center;
      Vector2d c_1 = E_1.center() - center;
      EE_0 = construct_from_shape_matrix(S_0, c_0);
      EE_1 = construct_from_shape_matrix(S_1, c_1);
      int num_inter = compute_intersection_points(inter_pts, EE_0, EE_1);

      for (int i = 0; i < num_inter; ++i)
        inter_pts[i] = delta.asDiagonal()*inter_pts[i] + center;
    /*}*/
#endif



    if (num_inter > 2)
    {
      Detail::PtCotg work[4];
      Detail::sort_points_by_polar_angle(inter_pts, work, num_inter);
    }

    // SPECIAL CASE.
    // If there is at most one intersection point, then either one of the
    // ellipse is included in the other.
    if (num_inter < 2)
    {
      if (E_0.contains(E_1.center()) || E_1.contains(E_0.center()))
        return std::min(area(E_0), area(E_1));
    }

    // GENERIC CASE
    // Compute the relative orientation of the intersection points w.r.t. the
    // ellipse orientation.
    double o_0[4];
    double o_1[4];
    orientation(o_0, inter_pts, num_inter, E_0);
    orientation(o_1, inter_pts, num_inter, E_1);

    // Sum the segment areas.
    double area = 0;
    for (int i = 0, j = num_inter-1; i < num_inter; j=i++)
    {
      double theta_0 = o_0[j];
      double theta_1 = o_0[i];

      double psi_0 = o_1[j];
      double psi_1 = o_1[i];

      if (theta_0 > theta_1)
        theta_1 += 2*M_PI;

      if (psi_0 > psi_1)
        psi_1 += 2*M_PI;

      area += min(segment_area(E_0, theta_0, theta_1),
                  segment_area(E_1, psi_0, psi_1));
    }
    // If the number of the intersection > 2, add the area of the polygon
    // whose vertices are p[0], p[1], ..., p[numInter].
    if (num_inter > 2)
    {
      for (int i = 0, j = num_inter-1; i < num_inter; j=i++)
      {
        Matrix2d M;
        M.col(0) = inter_pts[j];
        M.col(1) = inter_pts[i];
        area += 0.5*M.determinant();
      }
    }

    return area;
  }

  double analytic_jaccard_similarity(const Ellipse& e1, const Ellipse& e2)
  {
    double interArea = analytic_intersection(e1, e2);
    double unionArea = area(e1) + area(e2) - interArea;
    return interArea / unionArea;
  }


  // ======================================================================== //
  // Approximate computation of area of intersection ellipses.
  // ======================================================================== //
  std::vector<Point2d>
  discretize_ellipse(const Ellipse& e, int n)
  {
    std::vector<Point2d> polygon;
    polygon.reserve(n);

    const Matrix2d Ro(rotation2(e.orientation()));
    Vector2d D( e.radius1(), e.radius2() );

    for(int i = 0; i < n; ++i)
    {
      const double theta = 2.*M_PI*double(i)/n;
      const Matrix2d R(rotation2(theta));
      Point2d p(1.0, 0.0);

      const Point2d p1(e.center() + Ro.matrix()*D.asDiagonal()*R.matrix()*p);
      polygon.push_back(p1);
    }

    return polygon;
  }

  std::vector<Point2d>
  approximage_intersection(const Ellipse& e1, const Ellipse& e2, int n)
  {
    std::vector<Point2d> p1(discretize_ellipse(e1,n));
    std::vector<Point2d> p2(discretize_ellipse(e2,n));

    std::vector<Point2d> inter;
    inter = sutherland_hodgman(p1, p2);
    if (inter.empty())
      inter = sutherland_hodgman(p2, p1);
    return inter;
  }

  double approximate_jaccard_similarity(const Ellipse& e1, const Ellipse& e2,
                                        int n, double limit)
  {
    (void) limit;

    std::vector<Point2d> p1(discretize_ellipse(e1,n));
    std::vector<Point2d> p2(discretize_ellipse(e2,n));
    std::vector<Point2d> inter;
    inter = sutherland_hodgman(p1, p2);
    if (inter.empty())
      inter = sutherland_hodgman(p2, p1);

		if (!inter.empty())
		{
			double interArea = area(inter);
			double unionArea = area(p1)+area(p2)-interArea;
			return interArea/unionArea;
		}
		else
			return 0.;
  }

} /* namespace Sara */
} /* namespace DO */
