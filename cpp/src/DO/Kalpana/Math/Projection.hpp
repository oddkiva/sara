#pragma once

#include <DO/Kalpana/Defines.hpp>

#include <Eigen/Core>


namespace DO { namespace Kalpana {

  using namespace Eigen;

  DO_KALPANA_EXPORT
  Matrix4d frustum(double l, double r, double b, double t, double n, double f);

  DO_KALPANA_EXPORT
  Matrix4d perspective(double fov, double aspect, double z_near, double z_far);

  DO_KALPANA_EXPORT
  Matrix4d orthographic(double l, double r, double b, double t, double n,
                        double f);

  inline Matrix4d orthographic(double w, double h, double n, double f)
  {
    return orthographic(-w / 2, w / 2, -h / 2, h / 2, n, f);
  }

  DO_KALPANA_EXPORT
  auto look_at(const Vector3f& eye, const Vector3f& center, const Vector3f& up)
      -> Matrix4f;

} /* namespace Kalpana */
} /* namespace DO */
