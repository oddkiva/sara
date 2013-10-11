#pragma warning (disable : 4267)

//#define DEBUG_ELLIPSE_INTERSECTION

#include <DO/Geometry.hpp>
#include <DO/Graphics.hpp>

using namespace std;

namespace DO {

  // ========================================================================== //
  // Ellipse related functions
  // ========================================================================== //
  Matrix2d Ellipse::shapeMat() const
  {
    const Eigen::Rotation2D<double> R(o());
    Vector2d D( 1./(r1()*r1()), 1./(r2()*r2()) );
    return R.matrix()*D.asDiagonal()*R.matrix().transpose();
  }

  void Ellipse::drawOnScreen(const Color3ub col, double z) const
  {
    drawEllipse((c()*z).cast<float>(), z*r1(), z*r2(), toDegree(o()), col);
  }

  std::ostream& operator<<(std::ostream& os, const Ellipse& e)
  {
    os << "Ellipse info\n";
    os << "a = " << e.r1() << std::endl;
    os << "b = " << e.r2() << std::endl;
    os << "o = " << toDegree(e.o()) << " degree" << std::endl;
    os << "c = " << e.c().transpose() << std::endl;
    return os;
  }

  Ellipse fromShapeMat(const Matrix2d& shapeMat,
                       const Point2d& c)
  {
    Eigen::JacobiSVD<Matrix2d> svd(shapeMat, Eigen::ComputeFullU);
    const Vector2d r = svd.singularValues().cwiseSqrt().cwiseInverse();
    const Matrix2d& U = svd.matrixU();
    double o = std::atan2(U(1,0), U(0,0));
    return Ellipse(r(0), r(1), o, c);
  }

  // ========================================================================== //
  // Computation of polynomial roots
  // ========================================================================== //
  void printPolynomial(double p[], int d)
  {
    for(int i = 0; i <= d; ++i)
    {
      cout << p[i] << " X^" << i;
      if(i < d)
        cout << " + ";
    }
    cout << endl;
  }

  complex<double> computePolynomial(complex<double> x, double p[], int d)
  {
    complex<double> y = 0;
    for(int i = 0; i <= d; ++i)
    {
      complex<double> monomial = p[i]*pow(x,i);
      y += monomial;
    }
    return y;
  }

  void solveQuadraticEquation(bool& hasRealSolutions,
                              complex<double>& x1, complex<double>& x2,
                              double a, double b, double c)
  {
    double discriminant = b*b-4*a*c;
    x1 = (-b - sqrt(complex<double>(discriminant))) / (2*a);
    x2 = (-b + sqrt(complex<double>(discriminant))) / (2*a);
    if(discriminant >= 0)
      hasRealSolutions = true;
    else
      hasRealSolutions = false;
  }
  
  // Discriminant precision: 1e-3.
  void solveCubicEquation(complex<double>& z1, complex<double>& z2, complex<double>& z3,
                          double a, double b, double c, double d)
  {
    const double pi = 3.14159265358979323846;

    b /= a;
    c /= a;
    d /= a;
    a = 1.0;

    // Cardano's formula.
    const double p = (3*c-b*b)/3;
    const double q = (-9*c*b + 27*d + 2*b*b*b)/27;  
    const double delta = q*q + 4*p*p*p/27;

    if(delta < -1e-3)
    {
      const double theta = acos( -q/2*sqrt(27/(-p*p*p)) )/3.0;
      z1 = 2*sqrt(-p/3)*cos( theta );
      z2 = 2*sqrt(-p/3)*cos( theta + 2*pi/3);
      z3 = 2*sqrt(-p/3)*cos( theta + 4*pi/3);
    }
    else if(delta <= 1e-3)
    {
      z1 = 3*q/p;
      z2 = z3 = -3*q/(2*p);
    }
    else
    {
      double r1 = (-q+sqrt(delta))/2.0;
      double r2 = (-q-sqrt(delta))/2.0;
      double u = r1 < 0 ? -pow(-r1, 1.0/3.0) : pow(r1, 1.0/3.0);
      double v = r2 < 0 ? -pow(-r2, 1.0/3.0) : pow(r2, 1.0/3.0);
      complex<double> j(-0.5, sqrt(3.0)*0.5);
      z1 = u + v;
      z2 = j*u+conj(j)*v;
      z3 = j*j*u+conj(j*j)*v;
    }

    z1 -= b/(3*a);
    z2 -= b/(3*a);
    z3 -= b/(3*a);
  }
  
  // Involves the precision of the cubic equation solver: (1e-3.)
  void solveQuarticEquation(complex<double>& z1, complex<double>& z2,
                            complex<double>& z3, complex<double>& z4,
                            double a4, double a3, double a2, double a1, double a0)
  {
    a3 /= a4; a2/= a4; a1 /= a4; a0 /= a4; a4 = 1.0;

    double coeff[4];
    coeff[3] = 1.0;
    coeff[2] = -a2;
    coeff[1] = a1*a3 - 4.0*a0;
    coeff[0] = 4.0*a2*a0 - a1*a1 - a3*a3*a0;

    complex<double> y1, y2, y3;
    /*cout << "Intermediate cubic polynomial" << endl;
    printPolynomial(coeff, 3);*/
    solveCubicEquation(y1, y2, y3, coeff[3], coeff[2], coeff[1], coeff[0]);

    double yr = real(y1);
    double yi = fabs(imag(y1));
    if(yi > fabs(imag(y2)))
    {
      yr = real(y2);
      yi = fabs(imag(y2));
    }
    if(yi > fabs(imag(y3)))
    {
      yr = real(y3);
      yi = fabs(imag(y3));
    }

    complex<double> radicand = a3*a3/4.0 - a2 + yr;
    complex<double> R( sqrt(radicand) );
    complex<double> D, E;

    if(abs(R) > 1e-3)
    {
      D = sqrt( 3.0*a3*a3/4.0 - R*R - 2.0*a2 + (4.0*a3*a2 - 8.0*a1 - a3*a3*a3)/(4.0*R) );
      E = sqrt( 3.0*a3*a3/4.0 - R*R - 2.0*a2 - (4.0*a3*a2 - 8.0*a1 - a3*a3*a3)/(4.0*R) );
    }
    else
    {
      D = sqrt( 3.0*a3*a3/4.0 - 2.0*a2 + 2.0*sqrt(yr*yr - 4.0*a0) );
      E = sqrt( 3.0*a3*a3/4.0 - 2.0*a2 - 2.0*sqrt(yr*yr - 4.0*a0) );
    }

    z1 =  R/2.0 + D/2.0;
    z2 =  R/2.0 - D/2.0;
    z3 = -R/2.0 + E/2.0;
    z4 = -R/2.0 - E/2.0;

    // Check Viete's formula. 
    /*double p = a2 - 3*a3*a3/8;
    double q = a1 - a2*a3/2 + a3*a3*a3/8;
    double r = a0 - a1*a3/4 + a2*a3*a3/16 - 3*a3*a3*a3*a3/256;

    cout << "-2p = " << -2*p << endl;
    cout << pow(z1,2) + pow(z2,2) + pow(z3,2) + pow(z4,2) << endl;
    cout << "-3*q = " << -3*q << endl;
    cout << pow(z1,3) + pow(z2,3) + pow(z3,3) + pow(z4,3) << endl;
    cout << "2p^2 - 4r = " << 2*p*p - 4*r << endl;
    cout << pow(z1,4) + pow(z2,4) + pow(z3,4) + pow(z4,4) << endl;
    cout << "5pq = " << 5*p*q << endl;
    cout << pow(z1,5) + pow(z2,5) + pow(z3,5) + pow(z4,5) << endl;*/

    z1 -= a3/4;
    z2 -= a3/4;
    z3 -= a3/4;
    z4 -= a3/4;
  }

  void checkQuadraticEquationSolver()
  {
    // check quadratic equation solver
    bool hasRealSolutions;
    complex<double> x1, x2;
    double p[3] ={1.0, 0.0, 2.0};
    printPolynomial(p, 2);
    solveQuadraticEquation(hasRealSolutions, x1, x2, p[2], p[1], p[0]);
    cout << "x1 = " << x1 << " and x2 = " << x2 << endl;
    cout << "P(" << x1 << ") = " << computePolynomial(x1, p, 2) << endl;
    cout << "P(" << x2 << ") = " << computePolynomial(x2, p, 2) << endl;
    cout << endl;
  }

  void checkCubicEquationSolver()
  {
    // check quadratic equation solver
    complex<double> x1, x2, x3;
    for(int i = 0; i < 10; ++i)
    {
      cout << "iteration " << i << endl;
      double p[4] ={-rand()%10, -rand()%10, -rand()%10, rand()%10+1};
      printPolynomial(p, 3);

      solveCubicEquation(x1, x2, x3, p[3], p[2], p[1], p[0]);
      cout << "x1 = " << x1 << " and x2 = " << x2 << " and x3 = " << x3 << endl;
      cout << "|P(" << x1 << ")| = " << abs(computePolynomial(x1, p, 3)) << endl;
      cout << "|P(" << x2 << ")| = " << abs(computePolynomial(x2, p, 3)) << endl;
      cout << "|P(" << x3 << ")| = " << abs(computePolynomial(x3, p, 3)) << endl;
      cout << endl;
    }
  }

  void checkQuarticEquationSolver()
  {
    // check quadratic equation solver
    int realSolutionCount = 0;
    complex<double> x1, x2, x3, x4;
    for(int i = 0; i < 10; ++i)
    {
      cout << "iteration " << i << endl;
      double p[5] ={rand()%100000, rand()%10, rand()%10, rand()%10, rand()%10+1};
      printPolynomial(p, 4);
      solveQuarticEquation(x1, x2, x3, x4,
                           p[4], p[3], p[2], p[1], p[0]);

      cout << "x1 = " << x1 << " and x2 = " << x2 << endl;
      cout << "x3 = " << x3 << " and x4 = " << x4 << endl;
      cout << "|P(" << x1 << ")| = " << abs(computePolynomial(x1, p, 4)) << endl;
      cout << "|P(" << x2 << ")| = " << abs(computePolynomial(x2, p, 4)) << endl;
      cout << "|P(" << x3 << ")| = " << abs(computePolynomial(x3, p, 4)) << endl;
      cout << "|P(" << x4 << ")| = " << abs(computePolynomial(x4, p, 4)) << endl;
      cout << endl;
    }
  }

  // ========================================================================== //
  // Computation of intersecting points of intersecting ellipses
  // ========================================================================== //
  void printConicEquation(double s[6])
  {
    cout << s[0] << " + " << s[1] << " x + " << s[2] << " y + ";
    cout << s[3] << " x^2 + " << s[4] << " xy + " << s[5] << " y^2 = 0" << endl;
  }

  void getConicEquation(double s[6], const Ellipse & e)
  {
    const Matrix2d M(e.shapeMat());
    s[0] = e.c().x()*M(0,0)*e.c().x() + 2*e.c().x()*M(0,1)*e.c().y() + e.c().y()*M(1,1)*e.c().y() - 1.0;
    s[1] =-2.0*(M(0,0)*e.c().x() + M(0,1)*e.c().y());
    s[2] =-2.0*(M(1,0)*e.c().x() + M(1,1)*e.c().y());
    s[3] = M(0,0);
    s[4] = 2.0*M(0,1);
    s[5] = M(1,1);
  }

  void getQuarticEquation(double u[5], const double s[6], const double t[6])
  {
    double d[6][6];
    for(int i = 0; i < 6; ++i)
      for(int j = 0; j < 6; ++j)
        d[i][j] = s[i]*t[j] - s[j]*t[i];

    u[0] = d[3][1]*d[1][0] - d[3][0]*d[3][0];
    u[1] = d[3][4]*d[1][0] + d[3][1]*(d[4][0]+d[1][2]) - 2*d[3][2]*d[3][0];
    u[2] = d[3][4]*(d[4][0]+d[1][2]) + d[3][1]*(d[4][2]-d[5][1]) - d[3][2]*d[3][2] - 2*d[3][5]*d[3][0];
    u[3] = d[3][4]*(d[4][2]-d[5][1]) + d[3][1]*d[4][5] - 2*d[3][5]*d[3][2];
    u[4] = d[3][4]*d[4][5] - d[3][5]*d[3][5];
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
    if(imag(y) > 1e-6)
      return make_pair(false, Point2d());
    const double realY = real(y);
    double sigma[3], tau[3];
    getSigmaPolynomial(sigma, s, realY);
    getSigmaPolynomial(tau, t, realY);
    const double x = (sigma[2]*tau[0] - sigma[0]*tau[2])
                   / (sigma[1]*tau[2] - sigma[2]*tau[1]);

    if (fabs(computeConicExpression(x, realY, s)) < 5e-3 &&
        fabs(computeConicExpression(x, realY, t)) < 5e-3)
    {
      /*cout << "QO(x,y) = " << computeConicExpression(x, realY, s) << endl;
      cout << "Q1(x,y) = " << computeConicExpression(x, realY, t) << endl;*/
      return make_pair(true, Point2d(x,realY));
    }
    else
      return make_pair(false, Point2d());
  }

  void getEllipseIntersections(Point2d intersections[4], int& numInter,
                               const Ellipse & e1, const Ellipse & e2)
  {
    double s[6], t[6];
    double u[5];
    getConicEquation(s, e1);
    getConicEquation(t, e2);
    getQuarticEquation(u, s, t);

    complex<double> y[4];
    solveQuarticEquation(y[0], y[1], y[2], y[3],
                         u[4], u[3], u[2], u[1], u[0]);

    numInter = 0;
    for(int i = 0; i < 4; ++i)
    {
      pair<bool, Point2d> p(isRootValid(y[i], s, t));
      if(!p.first)
        continue;
      intersections[numInter] = p.second;
      ++numInter;
    }
  }

  // ========================================================================== //
  // Computation of the area of intersecting ellipses.
  // ========================================================================== //
  int findFirstPoint(const Point2d pts[], int numPts)
  {
    assert(numPts <= 4);

    Point2d first(pts[0]);
    int indexFirst = 0;
    for (int i = 1; i < numPts; ++i)
    {
      if (pts[i].y() < first.y())
      {
        first = pts[i];
        indexFirst = i;
      }
      else if (pts[i].y() == first.y() && pts[i].x() < first.x())
      {
        first = pts[i];
        indexFirst = 0;
      }
    }
    return indexFirst;
  }

  void ccwSortPoints(Point2d pts[], int numPts)
  {
    assert(numPts <= 4);

    int indexFirst = findFirstPoint(pts, numPts);
    std::swap(pts[0], pts[indexFirst]);

    // Compute polar angles with respect to the first point.
    std::pair<Point2d, double> ptsAngles[4];
    for (int i = 0; i < numPts; ++i)
    {
      if (i == 0)
        ptsAngles[i] = std::make_pair(pts[i], 0);
      else
        ptsAngles[i] = std::make_pair(pts[i],
        atan2(pts[i].y()-pts[0].y(), pts[i].x()-pts[0].x()));
    }
    // Sort points by polar angles.
    struct SortByAngle
    {
      typedef std::pair<Point2d, double> PtAngle;
      bool operator()(const PtAngle& a, const PtAngle& b) const
      { return a.second < b.second; }
    };
    std::sort(ptsAngles+1, ptsAngles+numPts, SortByAngle());

    // Recopy the sorted points.
    for (int i = 0; i < numPts; ++i)
      pts[i] = ptsAngles[i].first;
  }

  double convexSectorArea(const Ellipse& e, const Point2d pts[])
  {
    double theta[2];
    for (int i = 0; i < 2; ++i)
    {
      const Vector2d dir(pts[i]-e.c());
      double c = cos(e.o()), s = sin(e.o());
      const Vector2d u0( c, s);
      const Vector2d u1(-s, c);

      theta[i] = atan2(u1.dot(dir), u0.dot(dir));
    }

    if (abs(theta[1]-theta[0]) > M_PI)
    {
      if (theta[0] < 0)
        theta[0] += 2*M_PI;
      else
        theta[1] += 2*M_PI;
    }

    if (theta[0] > theta[1])
      std::swap(theta[0], theta[1]);

    return e.F(theta[1]) - e.F(theta[0]);
  }

  double analyticInterUnionRatio(const Ellipse& e1, const Ellipse& e2)
  {
    Point2d interPts[4];
    int numInter;
    getEllipseIntersections(interPts, numInter, e1, e2);
#ifdef DEBUG_ELLIPSE_INTERSECTION
    cout << "\nIntersection count = " << numInter << endl;
    for (int i = 0; i < numInter; ++i)
    {
      fillCircle(interPts[i].cast<float>(), 5.f, Green8);
      cout << "[" << i << "] " << interPts[i].transpose() << endl;
    }
#endif
    double interUnionRatio = 0.;
    if (numInter == 0 || numInter == 1)
    {
      // TODO: understand why there are numerical errors.
      // Namely, 'numInter' is actually '4' actually in some rare cases.
      double interArea = 0;
      double unionArea = e1.area()+e2.area();
      if (e1.isInside(e2.c()) || e2.isInside(e1.c()))
      {
        interArea = std::min(e1.area(), e2.area());
        unionArea = std::max(e1.area(), e2.area());
      }
      interUnionRatio = interArea/unionArea;
    }
    else if (numInter == 2)
    {
      // TODO: seems OK, I think in terms of numerical accuracy.
      Triangle t1(e1.c(), interPts[0], interPts[1]);
      Triangle t2(e2.c(), interPts[0], interPts[1]);
#ifdef DEBUG_ELLIPSE_INTERSECTION
      t1.drawOnScreen(Red8);
      t2.drawOnScreen(Blue8);
#endif
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
#ifdef DEBUG_ELLIPSE_INTERSECTION
      for (int i = 0; i < 2; ++i)
        cout << "Revert[" << i << "] = " << int(revert[i]) << endl;
#endif
      double ellSectArea1 = convexSectorArea(e1, interPts);
      double triArea1 = t1.area();
      double portionArea1 = ellSectArea1 - triArea1;
      if (revert[0])
        portionArea1 = e1.area() - portionArea1;
#ifdef DEBUG_ELLIPSE_INTERSECTION
      cout << "Ellipse 1" << endl;
      cout << "sectorArea1 = " << ellSectArea1 << endl;
      cout << "area1 = " << e1.area() << endl;
      cout << "triangleArea1 = " << triArea1 << endl;
      cout << "portionArea1 = " << portionArea1 << endl;
      cout << "portionAreaPercentage1 = " << portionArea1/e1.area() << endl;
#endif
      double ellSectArea2 = convexSectorArea(e2, interPts);
      double triArea2 = t2.area();
      double portionArea2 = ellSectArea2 - triArea2;
      if (revert[1])
        portionArea2 = e2.area() - portionArea2;
#ifdef DEBUG_ELLIPSE_INTERSECTION
      cout << "Ellipse 2" << endl;
      cout << "sectorArea2 = " << ellSectArea2 << endl;
      cout << "area2 = " << e2.area() << endl;
      cout << "triangleArea2 = " << triArea2 << endl;
      cout << "portionArea2 = " << portionArea2 << endl;
      cout << "portionAreaPercentage2 = " << portionArea2/e2.area() << endl;
#endif
      double interArea = portionArea1 + portionArea2;
      double unionArea = e1.area() + e2.area() - interArea;
      interUnionRatio = interArea/unionArea;
    }
    else // if (numInter == 3 || numInter == 4)
    {
      ccwSortPoints(interPts, numInter);
#ifdef DEBUG_ELLIPSE_INTERSECTION
      for (int i = 0; i < numInter; ++i)
        drawString(interPts[i].x()+10, interPts[i].y()+10, toString(i), Green8);
#endif
      double interArea = 0;
      // Add elliptic sectors.
      double ellipticSectorsArea = 0;
      for (int i = 0; i < numInter; ++i)
      {
        Point2d pts[2] = { interPts[i], interPts[(i+1)%numInter] };
        Triangle t1(e1.c(), pts[0], pts[1]);
        Triangle t2(e2.c(), pts[0], pts[1]);

        double sectorArea1 = convexSectorArea(e1, pts);
        double sectorArea2 = convexSectorArea(e2, pts);
        double triArea1 = t1.area();
        double triArea2 = t2.area();
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
      interArea = ellipticSectorsArea + quadArea;
      double unionArea = e1.area() + e2.area() - interArea;
      interUnionRatio = interArea/unionArea;
    }

    return interUnionRatio;
  }
}