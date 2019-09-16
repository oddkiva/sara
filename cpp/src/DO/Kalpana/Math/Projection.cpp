#include <DO/Kalpana/Math/Projection.hpp>
#include <Eigen/Geometry>


namespace DO { namespace Kalpana {

  Matrix4d frustum(double l, double r, double b, double t, double n, double f)
  {
    auto proj = Matrix4d{};

    proj <<
      2*n/(r-l),         0,  (r+l)/(r-l),            0,
              0, 2*n/(t-b),  (t+b)/(t-b),            0,
              0,         0, -(f+n)/(f-n), -2*f*n/(f-n),
              0,         0,           -1,            0;

    return proj;
  }

  Matrix4d perspective(double fov, double aspect, double z_near, double z_far)
  {
    auto t =  z_near * double(std::tan(fov * M_PI / 360.));
    auto b = -t;
    auto l = aspect * b;
    auto r = aspect * t;
    return frustum(l, r, b, t, z_near, z_far);
  }

  Matrix4d orthographic(double l, double r, double b, double t, double n,
                        double f)
  {
    auto proj = Matrix4d{};
    proj <<
      2/(r-l),       0,       0, -(r+l)/(r-l),
            0, 2/(t-b),       0, -(t+b)/(t-b),
            0,       0,-2/(f-n), -(f+n)/(f-n),
            0,       0,       0,            1;
    return proj;
  }

  auto look_at(const Vector3f& eye, const Vector3f& center, const Vector3f& up)
      -> Matrix4f
  {
    Vector3f f = (center - eye).normalized();
    Vector3f u = up.normalized();
    Vector3f s = f.cross(u).normalized();
    u = s.cross(f);

    Matrix4f res;
    res << s.x(), s.y(), s.z(), -s.dot(eye), u.x(), u.y(), u.z(), -u.dot(eye),
        -f.x(), -f.y(), -f.z(), f.dot(eye), 0, 0, 0, 1;

    return res;
  }

} /* namespace Kalpana */
} /* namespace DO */
