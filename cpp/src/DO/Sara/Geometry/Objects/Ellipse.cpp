#include <deque>

#include <DO/Sara/Geometry/Tools/Utilities.hpp>
#include <DO/Sara/Geometry/Objects/Ellipse.hpp>
#include <DO/Sara/Geometry/Objects/Triangle.hpp>
#include <DO/Sara/Geometry.hpp>


using namespace std;


namespace DO { namespace Sara {

  Vector2d Ellipse::rho(double theta) const
  {
    const Vector2d u{ unit_vector2(theta) };
    const auto& c = u(0);
    const auto& s = u(1);
    double r = (_a*_b) / sqrt(_b*_b*c*c + _a*_a*s*s);
    return r*u;
  }

  Point2d Ellipse::operator()(double theta) const
  {
    return _c + rotation2(_o)*rho(theta);
  }

  double orientation(const Point2d& p, const Ellipse& e)
  {
    const Vector2d x{ p - e.center() };
    const Vector2d u{ unit_vector2(e.orientation()) };
    const Vector2d v{ -u(1), u(0) };
    return atan2(v.dot(x), u.dot(x));
  }

  double segment_area(const Ellipse& e, double theta0, double theta1)
  {
    const Point2d p0{ e(theta0) };
    const Point2d p1{ e(theta1) };
    Triangle t{ e.center(), p0, p1 };

    const auto triangle_area = area(t);
    const auto sect_area = sector_area(e, theta0, theta1);

    if (abs(theta1 - theta0) < M_PI)
      return sect_area - triangle_area;
    return sect_area + triangle_area;
  }

  std::ostream& operator<<(std::ostream& os, const Ellipse& e)
  {
    os << "a = " << e.radius1() << std::endl;
    os << "b = " << e.radius2() << std::endl;
    os << "o = " << to_degree(e.orientation()) << " degree" << std::endl;
    os << "c = " << e.center().transpose() << std::endl;
    return os;
  }

  Ellipse construct_from_shape_matrix(const Matrix2d& shape_matrix, const Point2d& c)
  {
    Eigen::JacobiSVD<Matrix2d> svd(shape_matrix, Eigen::ComputeFullU);
    const Vector2d r = svd.singularValues().cwiseSqrt().cwiseInverse();
    const Matrix2d& U = svd.matrixU();
    double o = std::atan2(U(1,0), U(0,0));
    return Ellipse(r(0), r(1), o, c);
  }

  Quad oriented_bbox(const Ellipse& e)
  {
    const Vector2d u{ e.radius1(), e.radius2() };
    const BBox bbox{ -u, u };
    Quad quad{ bbox };
    Matrix2d R{ rotation2(e.orientation()) };
    for (int i = 0; i < 4; ++i)
      quad[i] = e.center() + R*quad[i];
    return quad;
  }

} /* namespace Sara */
} /* namespace DO */
