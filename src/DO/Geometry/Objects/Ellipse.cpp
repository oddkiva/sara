#pragma warning (disable : 4267)

//#define DEBUG_ELLIPSE_INTERSECTION

#include <DO/Geometry/Tools/Utilities.hpp>
#include <DO/Geometry/Objects/Ellipse.hpp>
#include <DO/Geometry/Objects/Triangle.hpp>
#include <DO/Geometry.hpp>
#include <deque>

using namespace std;

namespace DO {

  Point2d Ellipse::operator()(double theta) const 
  {
    Point2d p(unitVector2(theta));
    p(0) *= a_;
    p(1) *= b_;
    p += c_;
    return p;
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
    
    return polarAntiderivative(e, theta[1]) - polarAntiderivative(e, theta[0]);
  }

  double algebraicArea(const Ellipse& e, double theta0, double theta1)
  {
    return polarAntiderivative(e, theta1) - polarAntiderivative(e, theta0);
  }

  static
  void shiftOrientations(double& theta0, double& theta1) 
  {
    /*if ( theta0 < -M_PI || theta0 > M_PI ||
         theta1 < -M_PI || theta1 > M_PI )
    {
      const char *msg = "theta0 and theta1 must be in the range [-Pi, Pi]";
      throw std::logic_error(msg);
    }*/

    if (theta0 < 0)
      theta0 += 2*M_PI;
    if (theta1 < 0)
      theta1 += 2*M_PI;
  }

  double sectorArea(const Ellipse& e, double theta0, double theta1)
  {
    shiftOrientations(theta0, theta1);
    if (theta0 < theta1)
      return algebraicArea(e, theta0, theta1);
    else // theta0 > theta1
      return area(e) - algebraicArea(e, theta1, theta0);
  }

  double segmentArea(const Ellipse& e, double theta0, double theta1)
  {
    Point2d p0(e(theta0)), p1(e(theta1));

    Triangle t(e.c(), p0, p1);

    double triArea = area(t);
    double sectArea = sectorArea(e, theta0, theta1);

    if (abs(theta1 - theta0) < M_PI)
      return sectArea - triArea;
    return sectArea + triArea;
  }

  std::ostream& operator<<(std::ostream& os, const Ellipse& e)
  {
    os << "a = " << e.r1() << std::endl;
    os << "b = " << e.r2() << std::endl;
    os << "o = " << toDegree(e.o()) << " degree" << std::endl;
    os << "c = " << e.c().transpose() << std::endl;
    return os;
  }

  Ellipse fromShapeMat(const Matrix2d& shapeMat, const Point2d& c)
  {
    Eigen::JacobiSVD<Matrix2d> svd(shapeMat, Eigen::ComputeFullU);
    const Vector2d r = svd.singularValues().cwiseSqrt().cwiseInverse();
    const Matrix2d& U = svd.matrixU();
    double o = std::atan2(U(1,0), U(0,0));
    return Ellipse(r(0), r(1), o, c);
  }

}
