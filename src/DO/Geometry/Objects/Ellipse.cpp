#include <DO/Geometry/Tools/Utilities.hpp>
#include <DO/Geometry/Objects/Ellipse.hpp>
#include <DO/Geometry/Objects/Triangle.hpp>
#include <DO/Geometry.hpp>
#include <deque>

using namespace std;

namespace DO {

  Vector2d Ellipse::rho(double theta) const 
  {
    Vector2d u(unitVector2(theta));
    double& c = u(0);
    double& s = u(1);
    double r = (a_*b_) / sqrt(b_*b_*c*c + a_*a_*s*s);
    return r*u;
  }

  Point2d Ellipse::operator()(double theta) const 
  {
    return c_ + rotation2(o_)*rho(theta);
  }

  double orientation(const Point2d& p, const Ellipse& e)
  {
    const Vector2d x(p-e.center());
    const Vector2d u(unitVector2(e.orientation()));
    const Vector2d v(-u(1), u(0));
    return atan2(v.dot(x), u.dot(x));
  }

  double segmentArea(const Ellipse& e, double theta0, double theta1)
  {
    Point2d p0(e(theta0)), p1(e(theta1));

    Triangle t(e.center(), p0, p1);

    double triArea = area(t);
    double sectArea = sectorArea(e, theta0, theta1);

    if (abs(theta1 - theta0) < M_PI)
      return sectArea - triArea;
    return sectArea + triArea;
  }

  std::ostream& operator<<(std::ostream& os, const Ellipse& e)
  {
    os << "a = " << e.radius1() << std::endl;
    os << "b = " << e.radius2() << std::endl;
    os << "o = " << toDegree(e.orientation()) << " degree" << std::endl;
    os << "c = " << e.center().transpose() << std::endl;
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
