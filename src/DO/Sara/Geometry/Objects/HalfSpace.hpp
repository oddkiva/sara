#pragma once

namespace DO { namespace Sara {

  class HalfSpace3
  {
    Vector3d n0; // outward normal
    Point3d p0;  // some point in the plane.

  public:
    HalfSpace3() {}
    HalfSpace3(const Vector3d& normal, const Point3d& point)
      : n0(normal), p0(point) {}

    friend bool inside(const Point3d& p, const HalfSpace3& hs)
    { return hs.n0.dot(p-hs.p0) <= 0; }
  };

} /* namespace Sara */
} /* namespace DO */
