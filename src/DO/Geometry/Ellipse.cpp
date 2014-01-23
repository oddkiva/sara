#pragma warning (disable : 4267)

//#define DEBUG_ELLIPSE_INTERSECTION

#include <DO/Geometry.hpp>
#include <DO/Graphics.hpp>
#include <deque>

using namespace std;

namespace DO {
  
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
    
    return fabs(polarAntiderivative(e, theta[1]) - polarAntiderivative(e, theta[0]));
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
