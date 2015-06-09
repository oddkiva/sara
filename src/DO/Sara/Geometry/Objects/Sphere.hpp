#pragma once

namespace DO { namespace Sara {

  class Sphere
  {
    Point3d c_;
    double r_;
  public:
    Sphere(const Point3d& c, double r) : c_(c), r_(r) {}
    const Point3d& center() const { return c_; }
    double radius() const { return r_; }

    friend bool inside(const Point3d& x, const Sphere& S)
    { return (x - S.c_).squaredNorm() < S.radius()*S.radius(); }
  };

} /* namespace Sara */
} /* namespace DO */
