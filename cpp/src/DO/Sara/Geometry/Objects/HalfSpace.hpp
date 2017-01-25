#ifndef DO_SARA_GEOMETRY_OBJECTS_HALFSPACE_HPP
#define DO_SARA_GEOMETRY_OBJECTS_HALFSPACE_HPP

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  class HalfSpace3
  {
    //! @brief Outward normal.
    Vector3d _n0;
    //! @brief Some point in the plane.
    Point3d _p0;

  public:
    //! @{
    //! @brief Constructors.
    HalfSpace3() = default;

    HalfSpace3(const Vector3d& normal, const Point3d& point)
      : _n0{ normal }
      , _p0{ point }
    {
    }
    //! @}

    //! @brief Returns the normal of the plane.
    const Vector3d& normal() const
    {
      return _n0;
    }

    //! @brief Returns the reference point of the plane.
    const Vector3d& point() const
    {
      return _p0;
    }


    bool contains(const Point3d& p, double eps = 0.) const
    {
      return _n0.dot(p - _p0) <= eps;
    }
  };

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_GEOMETRY_OBJECTS_HALFSPACE_HPP */
