#pragma once

namespace DO { namespace Sara {

  class Cube
  {
    Point3d _a, _b;
  public:
    Cube() = default;

    Cube(Vector3d& origin, double side)
      : _a(origin), _b((origin.array()+side).matrix())
    {
    }

    Point3d const& a() const
    {
      return _a;
    }

    Point3d const& b() const
    {
      return _b;
    }

    friend bool inside(const Point3d& p, const Cube& cube)
    {
      return p.cwiseMin(cube._a) == cube._a && p.cwiseMax(cube._b) == cube._b;
    }

    friend double area(const Cube& c)
    {
      return std::pow((c._b - c._a)(0), 3);
    }
  };

} /* namespace Sara */
} /* namespace DO */
