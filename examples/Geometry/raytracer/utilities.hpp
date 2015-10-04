#pragma once

#include <DO/Sara/Core.hpp>


using namespace DO::Sara;


const double epsilon = 1e-10;

const Vector3d infinite_point(
  std::numeric_limits<double>::max(),
  std::numeric_limits<double>::max(),
  std::numeric_limits<double>::max());

const int w = 600;
const int h = 600;
const double f = 0;


inline Vector3d random_vector()
{
  auto vec = Vector3i{ rand(), rand(), rand() }.cast<double>();
  return vec / vec.norm();
}

inline Rgb64f random_color()
{
  return Color3i{ rand() % 256, rand() % 256, rand() % 256 }.cast<double>();
}

inline Vector3d random_reflection_properties()
{
  Vector3d rp = random_vector();
  rp.array() = rp.array().pow(2);
  return rp;
}

inline Matrix3d rotation_x(double theta)
{
  Matrix3d R;
  R <<
    1, 0, 0,
    0, cos(theta), -sin(theta),
    0, sin(theta), cos(theta);
  return R;
}

inline Matrix3d rotation_y(double theta)
{
  Matrix3d R;
  R <<
    cos(theta), 0, sin(theta),
    0, 1, 0,
    -sin(theta), 0, cos(theta);
  return R;
}

inline Matrix3d rotation_z(double theta)
{
  Matrix3d R;
  R <<
    cos(theta), -sin(theta), 0,
    sin(theta), cos(theta), 0,
    0, 0, 1;
  return R;
}

inline Matrix3d rotation(double alpha, double beta, double gamma)
{
  return rotation_z(gamma)*rotation_x(beta)*rotation_y(alpha);
}
