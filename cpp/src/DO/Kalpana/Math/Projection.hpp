#ifndef DO_KALPANA_MATH_PROJECTION_HPP
#define DO_KALPANA_MATH_PROJECTION_HPP

#include <Eigen/Core>


namespace DO { namespace Kalpana {

  using namespace Eigen;

  Matrix4d frustum(double l, double r, double b, double t, double n, double f);

  Matrix4d perspective(double fov, double aspect, double z_near, double z_far);

  Matrix4d orthographic(double l, double r, double b, double t,
                        double n, double f);

  inline
  Matrix4d orthographic(double w, double h, double n, double f)
  {
    return orthographic(-w/2, w/2, -h/2, h/2, n, f);
  }

} /* namespace Kalpana */
} /* namespace DO */


#endif /* DO_KALPANA_MATH_PROJECTION_HPP */