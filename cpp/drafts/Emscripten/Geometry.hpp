#pragma once

#include <Eigen/Core>

template <typename T>
inline auto frustum(T l, T r, T b, T t, T n, T f) -> Eigen::Matrix<T, 4, 4>
{
  auto proj = Eigen::Matrix<T, 4, 4>{};

  // clang-format off
  proj <<
    2*n/(r-l),         0,  (r+l)/(r-l),            0,
            0, 2*n/(t-b),  (t+b)/(t-b),            0,
            0,         0, -(f+n)/(f-n), -2*f*n/(f-n),
            0,         0,           -1,            0;
  // clang-format on

  return proj;
}

template <typename T>
inline auto orthographic(T l, T r, T b, T t, T n, T f) -> Eigen::Matrix<T, 4, 4>
{
  auto proj = Eigen::Matrix<T, 4, 4>{};

  // clang-format off
  proj <<
    2/(r-l),       0,  0, -(r+l)/(r-l),
          0, 2/(t-b),  0, -(t+b)/(t-b),
          0,       0,  0, -(f+n)/(f-n),
          0,       0,  0,            1;
  // clang-format on

  return proj;
}

template <typename T>
inline auto perspective(T fov, T aspect, T z_near, T z_far)
    -> Eigen::Matrix<T, 4, 4>
{
  static constexpr auto k = static_cast<T>(M_PI / 360.);
  const auto t = z_near * std::tan(fov * k);
  const auto b = -t;
  const auto l = aspect * b;
  const auto r = aspect * t;
  return frustum(l, r, b, t, z_near, z_far);
}

template <typename T>
inline auto look_at(const Eigen::Matrix<T, 3, 1>& eye,     //
                    const Eigen::Matrix<T, 3, 1>& center,  //
                    const Eigen::Matrix<T, 3, 1>& up) -> Eigen::Matrix<T, 4, 4>
{
  const Eigen::Matrix<T, 3, 1> f = (center - eye).normalized();
  Eigen::Matrix<T, 3, 1> u = up.normalized();
  const Eigen::Matrix<T, 3, 1> s = f.cross(u).normalized();
  u = s.cross(f);

  Eigen::Matrix<T, 4, 4> res;
  // clang-format off
  res <<
     s.x(),  s.y(),  s.z(), -s.dot(eye),
     u.x(),  u.y(),  u.z(), -u.dot(eye),
    -f.x(), -f.y(), -f.z(),  f.dot(eye),
         0,      0,      0,           1;
  // clang-format on

  return res;
}
