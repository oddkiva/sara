#pragma once

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Geometry.hpp>

#include "utilities.hpp"


using namespace DO::Sara;


// Default ambient coefficient
const double K_a = 0.2;

// Default diffuse coefficient
const double K_d = 0.5;

// Default specular coefficient
const double K_s = 0.3;
const double alpha = 8;

// Default reflection properties
const Vector3d default_reflection_properties{ K_a, K_d, K_s };


class GeometricObject
{
protected:
  Rgb64f _color;
  Vector3d _reflection_properties;

public:
  GeometricObject() = default;

  GeometricObject(const Rgb64f& c,
                  const Vector3d& reflection_properties
                    = default_reflection_properties)
    : _color{ c }
    , _reflection_properties{ reflection_properties }
  {
  }

  virtual ~GeometricObject()
  {
  }

  virtual bool hit(Point3d& impact,
                   Rgb64f& reflection,
                   const Point3d& camera_pos,
                   const Vector3d& ray,
                   const Point3d& light_pos) const = 0;

  virtual bool contains(const Point3d& p) const = 0;
};


class ObjectDifference : public GeometricObject
{
  const GeometricObject* _o1;
  const GeometricObject* _o2;

public:
  ObjectDifference(const GeometricObject& o1,
                   const GeometricObject& o2)
    : GeometricObject{}
    , _o1{ &o1 }
    , _o2{ &o2 }
  {
  }

  bool hit(Point3d& impact,
           Rgb64f& reflection,
           const Point3d& camera_pos,
           const Vector3d& ray,
           const Point3d& light_pos) const
  {
    auto impact1 = Point3d{};
    auto impact2 = Point3d{};
    auto ref1 = Rgb64f{};
    auto ref2 = Rgb64f{};

    auto hit1 = _o1->hit(impact1, ref1, camera_pos, ray, light_pos);
    auto hit2 = _o2->hit(impact2, ref2, camera_pos, ray, light_pos);

    impact = impact1;
    reflection = ref1;
    return hit1 && !hit2;
  }

  bool contains(const Point3d& p) const
  {
    return _o1->contains(p) && !_o2->contains(p);
  }
};


class ObjectIntersection : public GeometricObject
{
  const GeometricObject* _o1;
  const GeometricObject* _o2;

public:
  ObjectIntersection(const GeometricObject& o1,
                     const GeometricObject& o2)
    : GeometricObject{}
    , _o1{ &o1 }
    , _o2{ &o2 }
  {
  }

  bool hit(Point3d& impact, Rgb64f& reflection,
           const Point3d& camera_pos, const Vector3d& ray,
           const Point3d& light_pos) const
  {
    auto impact1 = Point3d{};
    auto impact2 = Point3d{};
    auto ref1 = Rgb64f{};
    auto ref2 = Rgb64f{};

    auto hit1 = _o1->hit(impact1, ref1, camera_pos, ray, light_pos);
    auto hit2 = _o2->hit(impact2, ref2, camera_pos, ray, light_pos);

    impact = impact1;
    reflection = ref1;

    return hit1 && hit2;
  }

  bool contains(const Point3d& p) const
  {
    return _o1->contains(p) && _o2->contains(p);
  }
};


class Sphere : public GeometricObject
{
  Point3d _c;
  double _r;

public:
  Sphere(const Point3d& c, double r, const Rgb64f& col);

  bool hit(Point3d& impact,
           Rgb64f& reflection,
           const Point3d& camera_pos,
           const Vector3d& ray,
           const Point3d& light_pos) const;

  bool contains(const Point3d& p) const
  {
    return (_c - p).squaredNorm() < _r*_r;
  }
};


class HalfSpace : public GeometricObject
{
  HalfSpace3 _hs;

public:
  HalfSpace() = default;

  HalfSpace(const Vector3d& n,
            const Vector3d& p,
            const Rgb64f& color)
    : GeometricObject{ color }
    , _hs{ n, p }
  {
  }

  bool hit(Point3d& impact,
           Rgb64f& reflection,
           const Point3d& camera_pos,
           const Vector3d& ray,
           const Point3d& light_pos) const;

  bool contains(const Point3d& p) const
  {
    return _hs.contains(p, epsilon);
  }
};

class Cube : public GeometricObject
{
  HalfSpace hs[6];

public:
  Cube(double side_size = 200.);

  bool hit(Point3d& impact,
           Rgb64f& reflection,
           const Point3d& camera_pos,
           const Vector3d& ray,
           const Point3d& light_pos) const;

  bool contains(const Point3d& p) const
  {
    for (int i = 0; i < 6; ++i)
      if (!hs[i].contains(p))
        return false;
    return true;
  }
};

// Difference of two objects
inline ObjectDifference operator-(const GeometricObject& o1,
                                  const GeometricObject& o2)
{
  return ObjectDifference(o1, o2);
}

// Intersection of two objects
inline ObjectIntersection operator*(const GeometricObject& o1,
                                    const GeometricObject& o2)
{
  return ObjectIntersection(o1, o2);
}
