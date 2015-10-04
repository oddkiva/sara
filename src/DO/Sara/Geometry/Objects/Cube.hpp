#ifndef DO_SARA_GEOMETRY_OBJECTS_CUBE_HPP
#define DO_SARA_GEOMETRY_OBJECTS_CUBE_HPP

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  class Cube
  {
    Point3d _a, _b;

  public:
    Cube() = default;

    Cube(Vector3d& origin, double side)
      : _a{ origin }
      , _b{ (origin.array() + side).matrix() }
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

    bool contains(const Point3d& p)
    {
      return p.cwiseMin(_a) == _a && p.cwiseMax(_b) == _b;
    }

    friend double area(const Cube& c)
    {
      return std::pow((c._b - c._a)(0), 3);
    }
  };

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_GEOMETRY_OBJECTS_CUBE_HPP */
