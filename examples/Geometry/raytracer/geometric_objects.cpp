#include <iostream>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Core.hpp>

#include "geometric_objects.hpp"


using namespace std;


bool HalfSpace::hit(Point3d& impact,
                    Rgb64f& reflections,
                    const Point3d& camera_pos,
                    const Vector3d& ray,
                    const Point3d& light_pos) const
{
  impact = infinite_point;
  reflections = Black64f;

  // Check that the ray is not collinear to the plane
  if (fabs(ray.dot(_hs.normal())) < epsilon)
    return false;

  const double t = -(camera_pos - _hs.point()).dot(_hs.normal()) / ray.dot(_hs.normal());

  // Wrong intersection way.
  if (t < epsilon)
    return false;

  // Compute the impact point.
  impact = camera_pos + t*ray;

  if (impact.z() < f + epsilon)
    return false;

  // Compute the reflected intensity value.
  Vector3d N = _hs.normal() / _hs.normal().norm();
  if (contains(light_pos))
    N *= -1.0;
  Vector3d L = (light_pos - impact) / (light_pos - impact).norm();

  Vector3d ambiant = _reflection_properties[0] * _color.cast<double>();
  Vector3d diffuse = _reflection_properties[1] * N.dot(L) * _color.cast<double>();
  Vector3d specular = _reflection_properties[2] * pow(N.dot(L), ::alpha) * _color.cast<double>();

  bool camera_in_half_space = _hs.contains(camera_pos);
  bool light_in_half_space = _hs.contains(light_pos);

  if ((camera_in_half_space && light_in_half_space) ||
    (!camera_in_half_space && !light_in_half_space))
    reflections = ambiant + diffuse + specular;
  else
    reflections = ambiant;

  return true;
}

::Sphere::Sphere(const Point3d& c, double r, const Rgb64f& col)
  : GeometricObject{ col }
  , _c{ c }
  , _r{ r }
{
}

bool ::Sphere::hit(Point3d& impact,
                   Rgb64f& reflections,
                   const Point3d& camera_pos,
                   const Vector3d& ray,
                   const Point3d& light_pos) const
{
  impact = infinite_point;
  reflections = Black64f;

  // Compute the closest impact point from the casted ray.
  double a = ray.squaredNorm();
  double b = 2. * ray.dot(camera_pos - _c);
  double c = (camera_pos - _c).squaredNorm() - _r*_r;
  double delta = b*b - 4 * a*c;
  if (delta < epsilon)
    return false;

  double t1 = (-b - sqrt(delta)) / (2 * a);
  double t2 = (-b + sqrt(delta)) / (2 * a);
  double t = min(t1, t2);
  impact = camera_pos + t*ray;

  // Compute the closest impact point from the light source
  Vector3d lightRay = impact - light_pos;
  double aa = lightRay.squaredNorm();
  double bb = 2. * lightRay.dot(light_pos - _c);
  double cc = (light_pos - _c).squaredNorm() - _r*_r;
  double delta2 = bb*bb - 4 * aa*cc;
  Point3d impact2 = infinite_point;
  if (delta2 >= epsilon)
  {
    double tt1 = (-bb - sqrt(delta2)) / (2 * aa);
    double tt2 = (-bb + sqrt(delta2)) / (2 * aa);
    double tt = min(tt1, tt2);
    impact2 = light_pos + tt*lightRay;
  }

  Vector3d N = (impact - _c) / (impact - _c).norm();
  Vector3d L = (light_pos - impact) / (light_pos - impact).norm();
  Vector3d R = (2. *N.dot(L) * N - L);

  const Vector3d ambiant = _reflection_properties[0] * _color.cast<double>();
  const Vector3d diffuse = _reflection_properties[1] * N.dot(L) * _color.cast<double>();
  const Vector3d specular = _reflection_properties[2] * pow((R.dot(N)), ::alpha) * _color.cast<double>();

  // Compute the reflected intensity value.
  if ((impact - impact2).squaredNorm() > epsilon*epsilon)
    reflections = ambiant;
  else
    reflections = ambiant + diffuse + specular;

  return true;
}

::Cube::Cube(double side_size)
{
  const double& d = side_size;
  const Point3d z0(0, 0, 500);

  Matrix3d R = rotation(0.2, -0.4, -0.8);

  // Y axis
  hs[3] = HalfSpace(R*Vector3d(0.02, 1, 0), d*Vector3d(0, 0.5, 0), Red64f);
  hs[4] = HalfSpace(R*Vector3d(0, -1, -0.03), d*Vector3d(0, -0.5, 0), Green64f);
  // X axis
  hs[0] = HalfSpace(R*Vector3d(1, -0.03, 0.03), d*Vector3d(0.5, 0, 0), Blue64f);
  hs[1] = HalfSpace(R*Vector3d(-1, -0.1, 0), d*Vector3d(-0.5, 0, 0), Yellow64f);
  // Z axis
  hs[5] = HalfSpace(R*Vector3d(0, 0.2, -1), d*Vector3d(0, 0, -0.5) + z0, Magenta64f);
  hs[2] = HalfSpace(R*Vector3d(0.1, 0, 1), d*Vector3d(0, 0, 0.5) + z0, Cyan64f);
}

bool ::Cube::hit(Point3d& impact,
                 Rgb64f& reflection,
                 const Point3d& camera_pos,
                 const Vector3d& ray,
                 const Point3d& light_pos) const
{
  impact = infinite_point;
  reflection = Black64f;

  const int n = 6;
  Point3d intersections[6];
  Rgb64f colors[6];
  bool plane_is_hit[6];
  bool impact_is_in_cube[6];

  for (int i = 0; i < n; ++i)
  {
    // Does the ray hit the i-th plane?
    plane_is_hit[i] = hs[i].hit(intersections[i], colors[i],
                                camera_pos, ray, light_pos);
    // Is the impact point in the cube?
    impact_is_in_cube[i] = contains(intersections[i]);
  }

  // The best impact point is the one that has the lowest depth value 'z'.
  for (int i = 0; i < n; ++i)
  {
    if (plane_is_hit[i] && impact_is_in_cube[i] &&
        intersections[i].z() < impact.z())
    {
      impact = intersections[i];
      reflection = colors[i];
    }
  }

  return impact != infinite_point;
}
