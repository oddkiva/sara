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

#include <DO/Geometry.hpp>
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
    s[0] = e.c().x()*M(0,0)*e.c().x() + 2*e.c().x()*M(0,1)*e.c().y() + e.c().y()*M(1,1)*e.c().y() - 1.0;
    s[1] =-2.0*(M(0,0)*e.c().x() + M(0,1)*e.c().y());
    s[2] =-2.0*(M(1,0)*e.c().x() + M(1,1)*e.c().y());
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
    if ( abs(imag(y)) > 1e-1*abs(real(y)) )
      return make_pair(false, Point2d());
    const double realY = real(y);
    double sigma[3], tau[3];
    getSigmaPolynomial(sigma, s, realY);
    getSigmaPolynomial(tau, t, realY);
    const double x = (sigma[2]*tau[0] - sigma[0]*tau[2])
                   / (sigma[1]*tau[2] - sigma[2]*tau[1]);

    if (fabs(computeConicExpression(x, realY, s)) < 1e-1 &&
        fabs(computeConicExpression(x, realY, t)) < 1e-1)
    {
      /*cout << "QO(x,y) = " << computeConicExpression(x, realY, s) << endl;
      cout << "Q1(x,y) = " << computeConicExpression(x, realY, t) << endl;*/
      return make_pair(true, Point2d(x,realY));
    }
    else
      return make_pair(false, Point2d());
  }
  
  struct Equal
  {
    Equal(double eps) : squared_eps_(eps*eps) {}
    bool operator()(const Point2d& p, const Point2d& q) const
    { return (p-q).squaredNorm() < squared_eps_; }
    double squared_eps_;
  };

  int computeEllipseIntersections(Point2d intersections[4],
                                  const Ellipse & e1, const Ellipse & e2)
  {
    double s[6], t[6];
    getConicEquation(s, e1);
    getConicEquation(t, e2);
    
    Polynomial<double, 4> u(getQuarticEquation(s, t));

    complex<double> y[4];
    roots(u, y[0], y[1], y[2], y[3]);

    int numInter = 0;
    for(int i = 0; i < 4; ++i)
    {
      pair<bool, Point2d> p(isRootValid(y[i], s, t));
      if(!p.first)
        continue;
      intersections[numInter] = p.second;
      ++numInter;
    }

    Equal equal(1e-2);
    return unique(intersections, intersections+numInter, equal) - intersections;
  }

  // ======================================================================== //
  // Computation of the area of intersecting ellipses.
  // ======================================================================== //
  double analyticInterArea(const Ellipse& e1, const Ellipse& e2, bool debug)
  {    
    Point2d interPts[4];
    int numInter = computeEllipseIntersections(interPts, e1, e2);

    if (debug)
    {
      cout << "\nIntersection count = " << numInter << endl;
      for (int i = 0; i < numInter; ++i)
      {
        fillCircle(interPts[i].cast<float>(), 5.f, Green8);
        cout << "[" << i << "] " << interPts[i].transpose() << endl;
      }
    }

    if (numInter < 2)
    {
      if (inside(e2.c(), e1) || inside(e1.c(), e2))
        return std::min(area(e1), area(e2));
    }
    
    if (numInter == 2)
    {
      // TODO: seems OK, I think in terms of numerical accuracy.
      Triangle t1(e1.c(), interPts[0], interPts[1]);
      Triangle t2(e2.c(), interPts[0], interPts[1]);
      if (debug)
      {
        drawTriangle(t1, Red8);
        drawTriangle(t2, Blue8);
      }
      
      // Find the correct elliptic sectors.
      bool revert[2] = {false, false};
      {
        const Point2d& c0 = e1.c();
        const Point2d& c1 = e2.c();
        const Point2d& p0 = interPts[0];
        const Point2d& p1 = interPts[1];

        Vector2d u(p1-p0);
        Vector2d n(-u(1), u(0));

        Vector2d dir0(c0-p0), dir1(c1-p0);
        double d0 = n.dot(dir0), d1 = n.dot(dir1);

        if (d0*d1 > 0)
        {
          if (abs(d0) < abs(d1))
            revert[0] = true;
          else
            revert[1] = true;
        }
      }

      if (debug)
        for (int i = 0; i < 2; ++i)
          cout << "Revert[" << i << "] = " << int(revert[i]) << endl;

      double ellSectArea1 = convexSectorArea(e1, interPts);
      double triArea1 = area(t1);
      double portionArea1 = ellSectArea1 - triArea1;
      if (revert[0])
        portionArea1 = area(e1) - portionArea1;

      if (debug)
      {
        cout << "Ellipse 1" << endl;
        cout << "sectorArea1 = " << ellSectArea1 << endl;
        cout << "area1 = " << area(e1) << endl;
        cout << "triangleArea1 = " << triArea1 << endl;
        cout << "portionArea1 = " << portionArea1 << endl;
        cout << "portionAreaPercentage1 = " << portionArea1/area(e1) << endl;
      }
      
      double ellSectArea2 = convexSectorArea(e2, interPts);
      double triArea2 = area(t2);
      double portionArea2 = ellSectArea2 - triArea2;
      if (revert[1])
        portionArea2 = area(e2) - portionArea2;

      if (debug)
      {
        cout << "Ellipse 2" << endl;
        cout << "sectorArea2 = " << ellSectArea2 << endl;
        cout << "area2 = " << area(e2) << endl;
        cout << "triangleArea2 = " << triArea2 << endl;
        cout << "portionArea2 = " << portionArea2 << endl;
        cout << "portionAreaPercentage2 = " << portionArea2/area(e2) << endl;
      }
      
      return portionArea1 + portionArea2;
    }
    
    if (numInter >= 3)
    {
      internal::PtCotg workArray[4];
      internal::sortPointsByPolarAngle(interPts, workArray, numInter);
      if (debug) {
        for (int i = 0; i < numInter; ++i)
          drawString(interPts[i].x()+10, interPts[i].y()+10, toString(i), Green8);
      }
      
      // Add elliptic sectors.
      double ellipticSectorsArea = 0;
      for (int i = 0; i < numInter; ++i)
      {
        Point2d pts[2] = { interPts[i], interPts[(i+1)%numInter] };
        Triangle t1(e1.c(), pts[0], pts[1]);
        Triangle t2(e2.c(), pts[0], pts[1]);
        
        drawTriangle(t1, Red8, 1);
        drawTriangle(t2, Blue8, 1);
        
        // correct here when pts[0], pts[1] more than pi degree.
        double sectorArea1 = convexSectorArea(e1, pts);
        double sectorArea2 = convexSectorArea(e2, pts);
        double triArea1 = area(t1);
        double triArea2 = area(t2);
        double portionArea1 = sectorArea1 - triArea1;
        double portionArea2 = sectorArea2 - triArea2;
        
        

        ellipticSectorsArea += std::min(portionArea1, portionArea2);
      }
      // Add inner quad area computed with the Green-Riemann formula.
      double quadArea = 0;
      for (int i = 0; i < numInter; ++i)
      {
        Matrix2d M;
        M.col(0) = interPts[i]; M.col(1) = interPts[(i+1)%numInter];
        quadArea += 0.5*M.determinant();
      }
      return ellipticSectorsArea + quadArea;
    }
    
    return 0;
  }
  
  double analyticInterUnionRatio(const Ellipse& e1, const Ellipse& e2)
  {
    double interArea = analyticInterArea(e1, e2);
    double unionArea = area(e1) + area(e2) - interArea;
    return interArea / unionArea;
  }

  // ======================================================================== //
  // Approximate computation of area of intersection ellipses.
  // ======================================================================== //
  std::vector<Point2d> discretizeEllipse(const Ellipse& e, int n)
  {
    std::vector<Point2d> polygon;
    polygon.reserve(n);
    
    const Matrix2d Ro(rotation2(e.o()));
    Vector2d D( e.r1(), e.r2() );
    
    for(int i = 0; i < n; ++i)
    {
      const double theta = 2.*M_PI*double(i)/n;
      const Matrix2d R(rotation2(theta));
      Point2d p(1.0, 0.0);
      
      const Point2d p1(e.c() + Ro.matrix()*D.asDiagonal()*R.matrix()*p);
      polygon.push_back(p1);
    }
    
    return polygon;
  }
  
  std::vector<Point2d> approxInter(const Ellipse& e1, const Ellipse& e2, int n)
  {
    std::vector<Point2d> p1(discretizeEllipse(e1,n));
    std::vector<Point2d> p2(discretizeEllipse(e2,n));

    std::vector<Point2d> inter;
    inter = sutherlandHodgman(p1, p2);
    if (inter.empty())
      inter = sutherlandHodgman(p2, p1);
    return inter;
  }
  
  double approximateIntersectionUnionRatio(const Ellipse& e1, const Ellipse& e2,
                                           int n, double limit)
  {
    typedef std::vector<Point2d> Polygon;
    
		if( !wellDefined(e1, limit) || !wellDefined(e2, limit) )
			return 0.;

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