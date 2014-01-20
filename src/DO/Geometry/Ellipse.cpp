#pragma warning (disable : 4267)

//#define DEBUG_ELLIPSE_INTERSECTION

#include <DO/Geometry.hpp>
#include <DO/Graphics.hpp>
#include <deque>

using namespace std;

namespace DO {

  std::ostream& operator<<(std::ostream& os, const Ellipse& e)
  {
    os << "Ellipse info\n";
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
