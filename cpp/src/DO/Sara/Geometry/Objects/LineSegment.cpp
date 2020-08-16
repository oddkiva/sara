#include <DO/Sara/Geometry/Objects/LineSegment.hpp>
#include <DO/Sara/Geometry/Tools/Utilities.hpp>


namespace DO { namespace Sara {

  bool intersection(const LineSegment& s1, const LineSegment& s2,
                    Point2d& u)
  {
    /*
      The intersection point 'u' is such that
      u = x0 + s(x1 - x0)  (1)
      u = y0 + t(y1 - y0)  (2)

      The goal is to determine the parameter 's' or 't'.
      It is sufficient to compute 's' for example.

      Using (1) = (2), it follows that
      s * (x1 - x0) - t * (y1 - y0) = y0 - x0

      Using cross-product with (y1-y0), we have
      s (x1-x0) x (y1-y0) = (y0-x0) x (y1-y0)
      Thus
      s = (y0-x0) x (y1-y0) / (x1-x0) x (y1-y0)
     */
    const auto dx = s1.direction();
    const auto dy = s2.direction();
    const Eigen::Vector2d d  = s2.p1() - s1.p1();

    // Sanity check: lines must not be collinear.
    const auto dxy = cross(dx, dy);
    if (fabs(dxy) < std::numeric_limits<double>::epsilon())
      return false;

    // Compute the parameter 's'.
    const auto s = cross(d, dy) / dxy;
    if (s < 0 || s > 1)
      return false;

    // Plug parameter 's' back to the equation (1).
    u = s1.p1() + s * dx;
    return true;
  }

} /* namespace Sara */
} /* namespace DO */
