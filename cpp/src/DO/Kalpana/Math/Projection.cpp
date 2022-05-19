#include <DO/Kalpana/Math/Projection.hpp>
#include <Eigen/Geometry>


namespace DO { namespace Kalpana {

  Matrix4d frustum(double l, double r, double b, double t, double n, double f)
  {
    auto proj = Matrix4d{};

    // clang-format off
    proj <<
      2*n/(r-l),         0,  (r+l)/(r-l),            0,
              0, 2*n/(t-b),  (t+b)/(t-b),            0,
              0,         0, -(f+n)/(f-n), -2*f*n/(f-n),
              0,         0,           -1,            0;
    // clang-format on

    return proj;
  }

  Matrix4d perspective(double fov, double aspect, double z_near, double z_far)
  {
    const auto t =  z_near * std::tan(fov * M_PI / 360.);
    const auto b = -t;
    const auto l = aspect * b;
    const auto r = aspect * t;
    return frustum(l, r, b, t, z_near, z_far);
  }

  Matrix4d orthographic(double l, double r, double b, double t, double n,
                        double f)
  {
    auto proj = Matrix4d{};
    // clang-format off
    proj <<
      2/(r-l),       0,       0, -(r+l)/(r-l),
            0, 2/(t-b),       0, -(t+b)/(t-b),
            0,       0,-2/(f-n), -(f+n)/(f-n),
            0,       0,       0,            1;
    // clang-format on
    return proj;
  }

  auto look_at(const Vector3f& eye, const Vector3f& center, const Vector3f& up)
      -> Matrix4f
  {
    const Vector3f f = (center - eye).normalized();
    Vector3f u = up.normalized();
    const Vector3f s = f.cross(u).normalized();
    u = s.cross(f);

    Matrix4f res;
    // clang-format off
    res <<
       s.x(),  s.y(),  s.z(), -s.dot(eye),
       u.x(),  u.y(),  u.z(), -u.dot(eye),
      -f.x(), -f.y(), -f.z(),  f.dot(eye),
           0,      0,      0,           1;
    // clang-format on

    return res;
  }

} /* namespace Kalpana */
} /* namespace DO */
