#pragma once

namespace DO { namespace Sara {

  class Cube
  {
    Point3d a_, b_;
  public:
    Cube() {}
    Cube(Vector3d& origin, double side)
      : a_(origin), b_((origin.array()+side).matrix()) {}

    Point3d const& a() const { return a_; }
    Point3d const& b() const { return b_; }

    friend bool inside(const Point3d& p, const Cube& cube)
    { return p.cwiseMin(cube.a_) == cube.a_ && p.cwiseMax(cube.b_) == cube.b_; }

    friend double area(const Cube& c)
    { return std::pow((c.b_ - c.a_)(0), 3); }
  };

} /* namespace Sara */
} /* namespace DO */
