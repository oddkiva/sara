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

#include <DO/Geometry/Algorithms/EllipseIntersection.hpp>
#include <DO/Geometry/Algorithms/ConvexHull.hpp>
#include <DO/Geometry/Algorithms/SutherlandHodgman.hpp>
#include <DO/Geometry/Tools/PolynomialRoots.hpp>
#include <DO/Geometry/Tools/Utilities.hpp>
#include <DO/Geometry/Graphics/DrawPolygon.hpp>
#include <DO/Core/Stringify.hpp>
#include <vector>
#include <iostream>

using namespace std;

namespace DO {

  // ======================================================================== //
  // Computation of intersecting points of intersecting ellipses
  // ======================================================================== //
  void printConicEquation(double s[6])
  {
    cout << s[0] << " + " << s[1] << " x + " << s[2] << " y + ";
    cout << s[3] << " x^2 + " << s[4] << " xy + " << s[5] << " y^2 = 0" << endl;
  }

  void getConicEquation(double s[6], const Ellipse & e)
  {
    const Matrix2d M(shapeMat(e));
    s[0] = e.center().x()*M(0,0)*e.center().x() + 2*e.center().x()*M(0,1)*e.center().y() + e.center().y()*M(1,1)*e.center().y() - 1.0;
    s[1] =-2.0*(M(0,0)*e.center().x() + M(0,1)*e.center().y());
    s[2] =-2.0*(M(1,0)*e.center().x() + M(1,1)*e.center().y());
    s[3] = M(0,0);
    s[4] = 2.0*M(0,1);
    s[5] = M(1,1);
  }

  Polynomial<double, 4> getQuarticEquation(const double s[6], const double t[6])
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

  void getSigmaPolynomial(double sigma[3], const double s[6], double y)
  {
    sigma[0] = s[0]+s[2]*y+s[5]*y*y;
    sigma[1] = s[1]+s[4]*y;
    sigma[2] = s[3];
  }

  inline double computeConicExpression(double x, double y, const double s[6])
  {
    return s[0] + s[1]*x + s[2]*y + s[3]*x*x + s[4]*x*y + s[5]*y*y;
  }

  // If imaginary part precision: 1e-6.
  // If the conic equation produces an almost zero value i.e.: 1e-3.
  // then we decide there is an intersection.
  pair<bool, Point2d> isRootValid(const complex<double>& y,
                                  const double s[6], const double t[6])
  {
    if ( abs(imag(y)) > 1e-2*abs(real(y)) )
      return make_pair(false, Point2d());
    const double realY = real(y);
    double sigma[3], tau[3];
    getSigmaPolynomial(sigma, s, realY);
    getSigmaPolynomial(tau, t, realY);
    const double x = (sigma[2]*tau[0] - sigma[0]*tau[2])
      / (sigma[1]*tau[2] - sigma[2]*tau[1]);

    if (fabs(computeConicExpression(x, realY, s)) < 1e-2 &&
        fabs(computeConicExpression(x, realY, t)) < 1e-2)
    {
      /*cout << "QO(x,y) = " << computeConicExpression(x, realY, s) << endl;
      cout << "Q1(x,y) = " << computeConicExpression(x, realY, t) << endl;*/
      return make_pair(true, Point2d(x,realY));
    }
    else
      return make_pair(false, Point2d());
  }

  int computeIntersectionPoints(Point2d intersections[4],
                                const Ellipse & e1, const Ellipse & e2)
  {
    // Rescale ellipse to try to improve numerical accuracy.
    Point2d center;
    center = 0.5*(e1.center() + e2.center());

    Ellipse ee1(e1), ee2(e2);
    ee1.center() -= center;
    ee2.center() -= center;

    double s[6], t[6];
    getConicEquation(s, ee1);
    getConicEquation(t, ee2);
    
    Polynomial<double, 4> u(getQuarticEquation(s, t));

    complex<double> y[4];
    roots(u, y[0], y[1], y[2], y[3]);


    int numInter = 0;
    for(int i = 0; i < 4; ++i)
    {
      pair<bool, Point2d> p(isRootValid(y[i], s, t));
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
    return it - intersections;
  }

  void orientation(double *ori, const Point2d *pts, int numPoints,
                   const Ellipse& e)
  {
    const Vector2d u(unitVector2(e.orientation()));
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
  double analyticIntersection(const Ellipse& E_0, const Ellipse& E_1)
  {
    // Find the intersection points of the two ellipses.
    Point2d interPts[4];
    int numInter = computeIntersectionPoints(interPts, E_0, E_1);
    if (numInter > 2)
    {
      internal::PtCotg work[4];
      internal::sortPointsByPolarAngle(interPts, work, numInter);
    }

    // SPECIAL CASE.
    // If there is at most one intersection point, then either one of the 
    // ellipse is included in the other.
    if (numInter < 2)
    {
      if (inside(E_1.center(), E_0) || inside(E_0.center(), E_1))
        return std::min(area(E_0), area(E_1));
    }

    // GENERIC CASE
    // Compute the relative orientation of the intersection points w.r.t. the 
    // ellipse orientation.
    double o_0[4];
    double o_1[4];
    orientation(o_0, interPts, numInter, E_0);
    orientation(o_1, interPts, numInter, E_1);

    // Sum the segment areas.
    double area = 0;
    for (int i = 0, j = numInter-1; i < numInter; j=i++)
    {
      double theta_0 = o_0[j];
      double theta_1 = o_0[i];

      double psi_0 = o_1[j];
      double psi_1 = o_1[i];

      if (theta_0 > theta_1)
        theta_1 += 2*M_PI;

      if (psi_0 > psi_1)
        psi_1 += 2*M_PI;

      area += min(segmentArea(E_0, theta_0, theta_1),
                  segmentArea(E_1, psi_0, psi_1));
    }
    // If the number of the intersection > 2, add the area of the polygon
    // whose vertices are p[0], p[1], ..., p[numInter].
    if (numInter > 2)
    {
      for (int i = 0, j = numInter-1; i < numInter; j=i++)
      {
        Matrix2d M;
        M.col(0) = interPts[j];
        M.col(1) = interPts[i];
        area += 0.5*M.determinant();
      }
    }

    return area;
  }
  
  double analyticJaccardSimilarity(const Ellipse& e1, const Ellipse& e2)
  {
    double interArea = analyticIntersection(e1, e2);
    double unionArea = area(e1) + area(e2) - interArea;
    return interArea / unionArea;
  }

  // ======================================================================== //
  // Approximate computation of area of intersection ellipses.
  // ======================================================================== //
  std::vector<Point2d>
  discretizeEllipse(const Ellipse& e, int n)
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
  approxIntersection(const Ellipse& e1, const Ellipse& e2, int n)
  {
    std::vector<Point2d> p1(discretizeEllipse(e1,n));
    std::vector<Point2d> p2(discretizeEllipse(e2,n));

    std::vector<Point2d> inter;
    inter = sutherlandHodgman(p1, p2);
    if (inter.empty())
      inter = sutherlandHodgman(p2, p1);
    return inter;
  }
  
  double approxJaccardSimilarity(const Ellipse& e1, const Ellipse& e2,
                                 int n, double limit)
  {
    std::vector<Point2d> p1(discretizeEllipse(e1,n));
    std::vector<Point2d> p2(discretizeEllipse(e2,n));
    std::vector<Point2d> inter;
    inter = sutherlandHodgman(p1, p2);
    if (inter.empty())
      inter = sutherlandHodgman(p2, p1);
    
		if (!inter.empty())
		{
			double interArea = area(inter);
			double unionArea = area(p1)+area(p2)-interArea;
			return interArea/unionArea;
		}
		else
			return 0.;
  }

} /* namespace DO */